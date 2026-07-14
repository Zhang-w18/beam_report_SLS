from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .codebook import BeamId
from .mcs import bler_from_sinr_db, select_mcs_from_sinr_db, tbs_bits_from_mcs
from .measurement import MeasurementResult
from .scheduler import ScheduleResult
from .utils import lin_to_db


@dataclass
class LinkEvalRow:
    scheme: str
    drop: int
    tti: int
    ue_id: int
    beam_index: int
    beam_id: str
    predicted_mcs: int
    actual_mcs: int
    effective_sinr_db: float
    tbler: float
    ack: int
    goodput_bits: int
    goodput_mbps: float
    olla_offset_db: float
    mcs_selection_sinr_db: float


def eesm(sinr_lin: np.ndarray, beta_db: float) -> float:
    beta = 10.0 ** (float(beta_db) / 10.0)
    x = np.asarray(sinr_lin, dtype=float)
    if x.size == 0:
        return 0.0
    z = -x / beta
    z_max = float(np.max(z))
    if not np.isfinite(z_max):
        return float("inf") if np.all(np.isposinf(x)) else 0.0
    # Stable log-mean-exp. The direct EESM formula underflows to log(0) for
    # very strong finite SINR grids, which produces spurious +inf in CSV output.
    val = -beta * (z_max + np.log(np.mean(np.exp(z - z_max))))
    return float(max(val, 0.0))


def realized_sinr_grid(schedule: ScheduleResult,
                       h_freq: np.ndarray,
                       tx_beams: np.ndarray,
                       rx_beams: np.ndarray,
                       beam_ids: Sequence[BeamId],
                       meas: MeasurementResult,
                       tx_power_w_per_panel: float,
                       ignore_interference: bool = False) -> Dict[int, np.ndarray]:
    """Return SINR[f] per scheduled UE using true H and selected RX beam."""
    out: Dict[int, np.ndarray] = {}
    links = schedule.links
    for link in links:
        u = link.ue_id
        m = link.beam_index
        bid_m = beam_ids[m]
        q = rx_beams[meas.selected_rx_beam[u, m]]
        h_sig = h_freq[u, bid_m.tx_unit]
        f_sig = tx_beams[m]
        hf = np.einsum("frt,t->fr", h_sig, f_sig)
        z = np.einsum("n,fn->f", np.conjugate(q), hf)
        sig = tx_power_w_per_panel * (np.abs(z) ** 2)
        den = np.full_like(sig, meas.noise_power_w, dtype=float)
        if not ignore_interference:
            for other in links:
                if other.ue_id == u:
                    continue
                bn = other.beam_index
                bid_n = beam_ids[bn]
                h_int = h_freq[u, bid_n.tx_unit]
                f_int = tx_beams[bn]
                hf_i = np.einsum("frt,t->fr", h_int, f_int)
                z_i = np.einsum("n,fn->f", np.conjugate(q), hf_i)
                den += tx_power_w_per_panel * (np.abs(z_i) ** 2)
        out[u] = sig / np.maximum(den, 1e-30)
    return out


def run_tti_loop(schedule: ScheduleResult,
                 h_freq: np.ndarray,
                 tx_beams: np.ndarray,
                 rx_beams: np.ndarray,
                 beam_ids: Sequence[BeamId],
                 meas: MeasurementResult,
                 tx_power_w_per_panel: float,
                 cfg: Dict,
                 drop_idx: int,
                 rng: np.random.Generator,
                 initial_olla: Dict[Tuple[str, int], float] | None = None,
                 link_adapter=None,
                 ignore_interference: bool = False) -> Tuple[List[LinkEvalRow], Dict[Tuple[str, int], float]]:
    num_tti = int(cfg["system"].get("num_tti_per_drop", 1))
    slot_ms = float(cfg["pdsch"].get("slot_duration_ms", 0.125))
    beta_db = float(cfg["link_abstraction"].get("eesm_beta_db", 5.0))
    slope = float(cfg["link_abstraction"].get("bler_curve_slope", 1.1))
    target = float(cfg["system"].get("target_bler", 0.1))
    olla_enabled = bool(cfg["link_abstraction"].get("olla_enabled", True))
    olla_step = float(cfg["link_abstraction"].get("olla_step_db", 0.1))

    olla = dict(initial_olla or {})
    rows: List[LinkEvalRow] = []
    sinr_grid = realized_sinr_grid(
        schedule, h_freq, tx_beams, rx_beams, beam_ids, meas,
        tx_power_w_per_panel, ignore_interference=ignore_interference,
    )

    for tti in range(num_tti):
        for link in schedule.links:
            key = (schedule.scheme, link.ue_id)
            off = float(olla.get(key, 0.0))
            sinr_eff_lin = eesm(sinr_grid[link.ue_id], beta_db=beta_db)
            eff_db = float(lin_to_db(sinr_eff_lin))
            # MCS selection must be causal: the scheduler only knows its
            # predicted SINR/CQI, not the realized post-scheduling SINR. The
            # realized effective SINR is used below only for BLER/ACK sampling.
            if olla_enabled:
                mcs_selection_sinr_db = float(link.predicted_sinr_db - off)
                if link_adapter is not None:
                    actual_mcs = int(link_adapter.select_mcs_from_sinr_db(mcs_selection_sinr_db))
                else:
                    actual_mcs = select_mcs_from_sinr_db(mcs_selection_sinr_db).index
            else:
                mcs_selection_sinr_db = float(link.predicted_sinr_db)
                actual_mcs = int(link.predicted_mcs)
            if link_adapter is not None:
                tbler = float(link_adapter.tbler_from_sinr_db(eff_db, actual_mcs))
            else:
                tbler = bler_from_sinr_db(eff_db, actual_mcs, slope=slope)
            ack = int(rng.uniform() > tbler)
            if ack:
                if link_adapter is not None:
                    goodput_bits = int(link_adapter.tbs_bits(actual_mcs))
                else:
                    goodput_bits = tbs_bits_from_mcs(
                        actual_mcs,
                        num_prbs=int(cfg["pdsch"]["num_prbs"]),
                        num_symbols=int(cfg["pdsch"]["num_symbols"]),
                        dmrs_overhead_re_per_prb=int(cfg["pdsch"].get("dmrs_overhead_re_per_prb", 0)),
                        num_layers=int(cfg["pdsch"].get("num_layers_per_ue", 1)),
                    )
            else:
                goodput_bits = 0
            goodput_mbps = goodput_bits / (slot_ms * 1e-3) / 1e6
            rows.append(LinkEvalRow(
                scheme=schedule.scheme,
                drop=drop_idx,
                tti=tti,
                ue_id=link.ue_id,
                beam_index=link.beam_index,
                beam_id=beam_ids[link.beam_index].short(),
                predicted_mcs=int(link.predicted_mcs),
                actual_mcs=int(actual_mcs),
                effective_sinr_db=eff_db,
                tbler=float(tbler),
                ack=ack,
                goodput_bits=int(goodput_bits),
                goodput_mbps=float(goodput_mbps),
                olla_offset_db=off,
                mcs_selection_sinr_db=mcs_selection_sinr_db,
            ))
            if olla_enabled:
                # off is a backoff subtracted from predicted_sinr_db.
                if ack:
                    off -= olla_step
                else:
                    off += olla_step * (1.0 - target) / max(target, 1e-6)
                olla[key] = float(off)
    return rows, olla
