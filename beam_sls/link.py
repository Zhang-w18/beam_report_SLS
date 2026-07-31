from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .codebook import ArrayConfig, BeamId, extract_panel_tx_dimension
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
    predicted_sinr_db: float
    predicted_mcs: int
    actual_mcs: int
    effective_sinr_db: float
    tbler: float
    ack: int
    goodput_bits: int
    goodput_mbps: float
    ack_random_uniform: float
    link_position: int
    olla_offset_db: float
    mcs_selection_sinr_db: float
    case_id: str | None = None
    feedback_scheme: str | None = None
    algorithm: str | None = None


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
                       ignore_interference: bool = False,
                       tx_array: ArrayConfig | None = None) -> Dict[int, np.ndarray]:
    """Return SINR[f] using full H and dynamically assigned physical panels."""
    out: Dict[int, np.ndarray] = {}
    links = schedule.links
    panel_by_link: Dict[Tuple[int, int], int] = {}
    next_panel_by_trp: Dict[Tuple[int, int], int] = {}
    for scheduled in links:
        bid = beam_ids[scheduled.beam_index]
        key = bid.trp_key()
        panel = int(next_panel_by_trp.get(key, 0))
        if tx_array is not None and panel >= int(tx_array.num_array_panels):
            raise ValueError(
                f"Schedule assigns more than {tx_array.num_array_panels} beams to TRP {key}"
            )
        panel_by_link[(scheduled.ue_id, scheduled.beam_index)] = panel
        next_panel_by_trp[key] = panel + 1

    def _panel_h(ue_id: int, beam_index: int) -> np.ndarray:
        bid = beam_ids[beam_index]
        h = h_freq[ue_id, bid.tx_unit]
        f = tx_beams[beam_index]
        if int(h.shape[-1]) == int(f.shape[-1]):
            return h
        if tx_array is None:
            raise ValueError(
                f"Full channel TX dimension {h.shape[-1]} and beam dimension "
                f"{f.shape[-1]} differ, but tx_array was not provided"
            )
        panel = panel_by_link[(ue_id, beam_index)]
        return extract_panel_tx_dimension(h, tx_array, panel)

    for link in links:
        u = link.ue_id
        m = link.beam_index
        bid_m = beam_ids[m]
        q = rx_beams[meas.selected_rx_beam[u, m]]
        h_sig = _panel_h(u, m)
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
                # The interferer's physical panel assignment is independent of
                # the victim UE. Look it up using the scheduled other link.
                h_int_full = h_freq[u, bid_n.tx_unit]
                f_int = tx_beams[bn]
                if int(h_int_full.shape[-1]) == int(f_int.shape[-1]):
                    h_int = h_int_full
                elif tx_array is not None:
                    h_int = extract_panel_tx_dimension(
                        h_int_full,
                        tx_array,
                        panel_by_link[(other.ue_id, other.beam_index)],
                    )
                else:
                    raise ValueError("tx_array is required for panel channel views")
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
                 ignore_interference: bool = False,
                 mcs_adapter=None) -> Tuple[List[LinkEvalRow], Dict[Tuple[str, int], float]]:
    if link_adapter is None:
        raise ValueError("run_tti_loop requires an explicit link-adaptation backend")
    if mcs_adapter is None:
        mcs_adapter = link_adapter
    num_tti = int(cfg["system"].get("num_tti_per_drop", 1))
    warmup_tti = int(cfg["link_abstraction"].get("olla_warmup_tti", 0))
    if warmup_tti < 0:
        raise ValueError("link_abstraction.olla_warmup_tti must be >= 0")
    slot_ms = float(cfg["pdsch"].get("slot_duration_ms", 0.125))
    beta_db = float(cfg["link_abstraction"].get("eesm_beta_db", 5.0))
    target = float(cfg["system"].get("target_bler", 0.1))
    olla_enabled = bool(cfg["link_abstraction"].get("olla_enabled", True))
    olla_step = float(cfg["link_abstraction"].get("olla_step_db", 0.1))

    olla = dict(initial_olla or {})
    rows: List[LinkEvalRow] = []
    sinr_grid = realized_sinr_grid(
        schedule, h_freq, tx_beams, rx_beams, beam_ids, meas,
        tx_power_w_per_panel, ignore_interference=ignore_interference,
        tx_array=(
            ArrayConfig.from_dict(cfg["tx_array"])
            if h_freq.shape[-1] != tx_beams.shape[-1]
            else None
        ),
    )
    links = list(schedule.links)
    effective_sinr_db = np.asarray([
        float(lin_to_db(eesm(sinr_grid[link.ue_id], beta_db=beta_db)))
        for link in links
    ], dtype=float)

    # Negative loop indices are warmup TTIs. They consume ACK randomness and
    # update OLLA exactly like measured TTIs, but are not written to link_tti.csv.
    # Measured rows retain the backward-compatible tti range [0, num_tti).
    for tti in range(-warmup_tti, num_tti):
        offsets = np.asarray([
            float(olla.get(
                (schedule.case_id or schedule.scheme, link.ue_id), 0.0,
            ))
            for link in links
        ], dtype=float)
        predicted_sinr_db = np.asarray(
            [float(link.predicted_sinr_db) for link in links], dtype=float,
        )
        mcs_selection_sinr_db = (
            predicted_sinr_db - offsets if olla_enabled else predicted_sinr_db
        )
        if hasattr(mcs_adapter, "select_mcs_from_sinr_db_batch"):
            actual_mcs = np.asarray(
                mcs_adapter.select_mcs_from_sinr_db_batch(
                    mcs_selection_sinr_db,
                ),
                dtype=np.int32,
            ).reshape(-1)
        elif hasattr(mcs_adapter, "map_sinr_db"):
            actual_mcs = np.asarray(
                mcs_adapter.map_sinr_db(mcs_selection_sinr_db)[0],
                dtype=np.int32,
            ).reshape(-1)
        else:
            actual_mcs = np.asarray([
                int(mcs_adapter.select_mcs_from_sinr_db(float(value)))
                for value in mcs_selection_sinr_db
            ], dtype=np.int32)
        if hasattr(link_adapter, "tbler_from_sinr_db_batch"):
            tbler_values = np.asarray(
                link_adapter.tbler_from_sinr_db_batch(
                    effective_sinr_db, actual_mcs,
                ),
                dtype=float,
            ).reshape(-1)
        else:
            tbler_values = np.asarray([
                float(link_adapter.tbler_from_sinr_db(float(sinr), int(mcs)))
                for sinr, mcs in zip(effective_sinr_db, actual_mcs)
            ], dtype=float)

        for link_position, link in enumerate(links):
            key = (schedule.case_id or schedule.scheme, link.ue_id)
            off = float(offsets[link_position])
            eff_db = float(effective_sinr_db[link_position])
            # MCS selection is causal: the scheduler only knows the reported /
            # predicted SINR and its OLLA state. The realized post-scheduling
            # SINR is simulator truth and is used only for TBLER/ACK evaluation.
            selection_db = float(mcs_selection_sinr_db[link_position])
            selected_mcs = int(actual_mcs[link_position])
            tbler = float(tbler_values[link_position])
            ack_random_uniform = float(rng.uniform())
            ack = int(ack_random_uniform > tbler)
            if ack:
                goodput_bits = int(link_adapter.tbs_bits(selected_mcs))
            else:
                goodput_bits = 0
            goodput_mbps = goodput_bits / (slot_ms * 1e-3) / 1e6
            if tti >= 0:
                rows.append(LinkEvalRow(
                    scheme=schedule.case_id or schedule.scheme,
                    drop=drop_idx,
                    tti=tti,
                    ue_id=link.ue_id,
                    beam_index=link.beam_index,
                    beam_id=beam_ids[link.beam_index].short(),
                    predicted_sinr_db=float(link.predicted_sinr_db),
                    predicted_mcs=int(link.predicted_mcs),
                    actual_mcs=selected_mcs,
                    effective_sinr_db=eff_db,
                    tbler=float(tbler),
                    ack=ack,
                    goodput_bits=int(goodput_bits),
                    goodput_mbps=float(goodput_mbps),
                    ack_random_uniform=ack_random_uniform,
                    link_position=int(link_position),
                    olla_offset_db=off,
                    mcs_selection_sinr_db=selection_db,
                    case_id=schedule.case_id or schedule.scheme,
                    feedback_scheme=schedule.feedback_scheme or schedule.scheme,
                    algorithm=schedule.algorithm or schedule.metadata.get("algorithm"),
                ))
            if olla_enabled:
                # Positive off is a backoff; a negative value boosts the
                # reported/predicted SINR. It is never applied to simulator
                # truth (effective_sinr_db).
                if ack:
                    off -= olla_step
                else:
                    off += olla_step * (1.0 - target) / max(target, 1e-6)
                olla[key] = float(off)
    return rows, olla


def run_one_tti(schedule: ScheduleResult,
                h_freq: np.ndarray,
                tx_beams: np.ndarray,
                rx_beams: np.ndarray,
                beam_ids: Sequence[BeamId],
                meas: MeasurementResult,
                tx_power_w_per_panel: float,
                cfg: Dict,
                drop_idx: int,
                tti: int,
                rng: np.random.Generator,
                initial_olla: Dict[Tuple[str, int], float] | None = None,
                link_adapter=None,
                ignore_interference: bool = False,
                record: bool = True,
                mcs_adapter=None) -> Tuple[List[LinkEvalRow], Dict[Tuple[str, int], float]]:
    """Evaluate one scheduled TTI and immediately update per-UE OLLA state."""
    one_tti_cfg = {
        **cfg,
        "system": {**cfg["system"], "num_tti_per_drop": 1},
        "link_abstraction": {
            **cfg["link_abstraction"],
            "olla_warmup_tti": 0,
        },
    }
    rows, olla = run_tti_loop(
        schedule, h_freq, tx_beams, rx_beams, beam_ids, meas,
        tx_power_w_per_panel, one_tti_cfg, drop_idx, rng, initial_olla,
        link_adapter=link_adapter, mcs_adapter=mcs_adapter,
        ignore_interference=ignore_interference,
    )
    for row in rows:
        row.tti = int(tti)
    return (rows if record else []), olla
