from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .codebook import BeamId
from .feedback import UEReport
from .mcs import rate_mbps_from_mcs, select_mcs_from_sinr_lin
from .utils import lin_to_db


@dataclass
class ScheduledLink:
    ue_id: int
    beam_index: int
    predicted_sinr_db: float
    predicted_mcs: int
    predicted_rate_mbps: float

    def to_dict(self, beam_ids: Sequence[BeamId]) -> Dict:
        return {
            "ue_id": self.ue_id,
            "beam_index": self.beam_index,
            "beam_id": beam_ids[self.beam_index].short(),
            "predicted_sinr_db": self.predicted_sinr_db,
            "predicted_mcs": self.predicted_mcs,
            "predicted_rate_mbps": self.predicted_rate_mbps,
        }


@dataclass
class ScheduleResult:
    scheme: str
    objective_value: float
    links: List[ScheduledLink]

    def to_dict(self, beam_ids: Sequence[BeamId]) -> Dict:
        return {
            "scheme": self.scheme,
            "objective_value": self.objective_value,
            "links": [l.to_dict(beam_ids) for l in self.links],
        }


def schedule(reports: List[UEReport],
             beam_ids: Sequence[BeamId],
             cfg: Dict,
             tbar_mbps: Dict[int, float] | None = None,
             link_adapter=None) -> ScheduleResult:
    alg = cfg["scheduler"].get("algorithm", "exhaustive")
    if alg == "exhaustive":
        return exhaustive_schedule(reports, beam_ids, cfg, tbar_mbps, link_adapter)
    if alg == "greedy":
        return greedy_schedule(reports, beam_ids, cfg, tbar_mbps, link_adapter)
    raise ValueError(f"Unknown scheduler.algorithm={alg}")


def _rate_kwargs(cfg: Dict) -> Dict:
    return {
        "num_prbs": int(cfg["pdsch"]["num_prbs"]),
        "num_symbols": int(cfg["pdsch"]["num_symbols"]),
        "dmrs_overhead_re_per_prb": int(cfg["pdsch"].get("dmrs_overhead_re_per_prb", 0)),
        "slot_duration_ms": float(cfg["pdsch"].get("slot_duration_ms", 0.125)),
        "num_layers": int(cfg["pdsch"].get("num_layers_per_ue", 1)),
    }


def _pf_weight(ue_id: int, cfg: Dict, tbar_mbps: Dict[int, float] | None) -> float:
    if cfg["scheduler"].get("objective", "sum_rate") != "proportional_fair":
        return 1.0
    init = float(cfg["scheduler"].get("pf_tbar_init_mbps", 1.0))
    return 1.0 / max(init if tbar_mbps is None else tbar_mbps.get(ue_id, init), 1e-6)


def _panel_constraint_ok(assignments: Sequence[Tuple[int, int]],
                         reports: List[UEReport],
                         beam_ids: Sequence[BeamId],
                         cfg: Dict) -> bool:
    if not cfg["scheduler"].get("use_panel_constraint", True):
        return True
    used = set()
    for ue_id, b in assignments:
        key = beam_ids[b].panel_key()
        if key in used:
            return False
        used.add(key)
    return True


def _evaluate_assignments(assignments: Sequence[Tuple[int, int]],
                          reports: List[UEReport],
                          beam_ids: Sequence[BeamId],
                          cfg: Dict,
                          tbar_mbps: Dict[int, float] | None,
                          link_adapter=None) -> Tuple[float, List[ScheduledLink]]:
    if not assignments:
        return 0.0, []

    rep_by_ue = {r.ue_id: r for r in reports}
    scheme = reports[0].scheme if reports else "unknown"
    rate_kwargs = _rate_kwargs(cfg)
    penalty_lambda = float(cfg["scheduler"].get("conflict_penalty_lambda", 0.0))

    links: List[ScheduledLink] = []
    utility = 0.0
    conflict_penalty = 0.0

    for ue_id, b in assignments:
        rep = rep_by_ue[ue_id]
        cand = rep.candidate_by_beam(b)
        if cand is None:
            return -np.inf, []

        if scheme == "full_gamma" and rep.full_gamma is not None and rep.full_service_power_w is not None:
            s = float(rep.full_service_power_w[b])
            den = float(rep.full_noise_power_w)
            for other_ue, other_b in assignments:
                if other_ue == ue_id:
                    continue
                # I = S/Gamma - N, clipped for numerical safety.
                g = float(rep.full_gamma[b, other_b])
                den += max(0.0, s / max(g, 1e-30) - float(rep.full_noise_power_w))
            pred_sinr_lin = s / max(den, 1e-30)
            if link_adapter is not None:
                mcs = int(link_adapter.select_mcs_from_sinr_lin(float(pred_sinr_lin)))
            else:
                mcs = select_mcs_from_sinr_lin(pred_sinr_lin).index
            pred_sinr_db = float(lin_to_db(pred_sinr_lin))
        else:
            # Limited-feedback scheduler uses SU MCS/rate and an ID-only conflict
            # penalty for proposed reports.
            mcs = int(cand.su_mcs)
            pred_sinr_db = float(cand.su_snr_db)
            for other_ue, other_b in assignments:
                if other_ue != ue_id and other_b in cand.conflict_beams:
                    conflict_penalty += 1.0

        if link_adapter is not None:
            r_mbps = float(link_adapter.rate_mbps(mcs))
        else:
            r_mbps = rate_mbps_from_mcs(mcs, **rate_kwargs)
        utility += _pf_weight(ue_id, cfg, tbar_mbps) * r_mbps
        links.append(ScheduledLink(ue_id=ue_id,
                                   beam_index=b,
                                   predicted_sinr_db=pred_sinr_db,
                                   predicted_mcs=mcs,
                                   predicted_rate_mbps=r_mbps))
    if scheme in ("topk_conflict_id", "threshold_conflict_set"):
        utility -= penalty_lambda * conflict_penalty
    return float(utility), links


def _effective_max_mu_order(cfg: Dict) -> int:
    resolved = cfg.get("_resolved", {}).get("max_mu_order", None)
    if resolved is not None:
        return int(resolved)
    raw = cfg["scheduler"].get("max_mu_order", 1)
    if raw is None or str(raw).lower() == "auto":
        return 1
    return int(raw)


def exhaustive_schedule(reports: List[UEReport],
                        beam_ids: Sequence[BeamId],
                        cfg: Dict,
                        tbar_mbps: Dict[int, float] | None = None,
                        link_adapter=None) -> ScheduleResult:
    max_q = _effective_max_mu_order(cfg)
    reports = [r for r in reports if r.candidates]
    best_val = 0.0
    best_links: List[ScheduledLink] = []
    scheme = reports[0].scheme if reports else "unknown"

    for q in range(1, min(max_q, len(reports)) + 1):
        for subset in combinations(reports, q):
            cand_lists = [[c.beam_index for c in r.candidates] for r in subset]
            for beams in product(*cand_lists):
                assignments = [(subset[i].ue_id, int(beams[i])) for i in range(q)]
                if not _panel_constraint_ok(assignments, reports, beam_ids, cfg):
                    continue
                val, links = _evaluate_assignments(assignments, reports, beam_ids, cfg, tbar_mbps, link_adapter)
                if val > best_val:
                    best_val = val
                    best_links = links
    return ScheduleResult(scheme=scheme, objective_value=float(best_val), links=best_links)


def greedy_schedule(reports: List[UEReport],
                    beam_ids: Sequence[BeamId],
                    cfg: Dict,
                    tbar_mbps: Dict[int, float] | None = None,
                    link_adapter=None) -> ScheduleResult:
    max_q = _effective_max_mu_order(cfg)
    reports = [r for r in reports if r.candidates]
    scheme = reports[0].scheme if reports else "unknown"
    current: List[Tuple[int, int]] = []
    current_val = 0.0
    used_ues = set()

    while len(current) < max_q:
        best_delta = 0.0
        best_assignment = None
        best_val = current_val
        for r in reports:
            if r.ue_id in used_ues:
                continue
            for c in r.candidates:
                trial = current + [(r.ue_id, c.beam_index)]
                if not _panel_constraint_ok(trial, reports, beam_ids, cfg):
                    continue
                val, _ = _evaluate_assignments(trial, reports, beam_ids, cfg, tbar_mbps, link_adapter)
                delta = val - current_val
                if delta > best_delta:
                    best_delta = delta
                    best_assignment = (r.ue_id, c.beam_index)
                    best_val = val
        if best_assignment is None:
            break
        current.append(best_assignment)
        used_ues.add(best_assignment[0])
        current_val = best_val
    final_val, final_links = _evaluate_assignments(current, reports, beam_ids, cfg, tbar_mbps, link_adapter)
    return ScheduleResult(scheme=scheme, objective_value=final_val, links=final_links)
