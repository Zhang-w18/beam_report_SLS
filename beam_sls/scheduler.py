from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .codebook import BeamId
from .feedback import UEReport
from .mcs import bler_from_sinr_db, rate_mbps_from_mcs, select_mcs_from_sinr_lin
from .utils import lin_to_db


GLOBAL_DOMAIN_MODES = {"global", "network", "global_joint", "network_joint"}
SECTOR_DOMAIN_MODES = {
    "sector",
    "cell",
    "per_sector",
    "per_cell",
    "sector_independent",
    "cell_independent",
    "per_sector_independent",
    "per_cell_independent",
    "single_site_three_sector_independent",
}
SITE_DOMAIN_MODES = {
    "site",
    "site_joint",
    "site_domain",
    "per_site",
    "per_site_joint",
    "per_site_three_sector_joint",
}


def normalize_domain_mode(mode: str | None) -> str:
    raw = str(mode or "global").lower()
    if raw in GLOBAL_DOMAIN_MODES:
        return "global"
    if raw in SECTOR_DOMAIN_MODES:
        return "per_sector_independent"
    if raw in SITE_DOMAIN_MODES:
        return "per_site_joint"
    raise ValueError(f"Unknown scheduler.domain_mode={mode}")


def is_site_domain_mode(mode: str | None) -> bool:
    return normalize_domain_mode(mode) == "per_site_joint"


def is_sector_domain_mode(mode: str | None) -> bool:
    return normalize_domain_mode(mode) == "per_sector_independent"


@dataclass
class ScheduledLink:
    ue_id: int
    beam_index: int
    predicted_sinr_db: float
    predicted_mcs: int
    predicted_rate_mbps: float
    predicted_outage: bool = False

    def to_dict(self, beam_ids: Sequence[BeamId]) -> Dict:
        return {
            "ue_id": self.ue_id,
            "beam_index": self.beam_index,
            "beam_id": beam_ids[self.beam_index].short(),
            "predicted_sinr_db": self.predicted_sinr_db,
            "predicted_mcs": self.predicted_mcs,
            "predicted_rate_mbps": self.predicted_rate_mbps,
            "predicted_outage": self.predicted_outage,
        }


@dataclass
class ScheduleResult:
    scheme: str
    objective_value: float
    links: List[ScheduledLink]
    metadata: Dict = field(default_factory=dict)
    case_id: str | None = None
    feedback_scheme: str | None = None
    algorithm: str | None = None

    def to_dict(self, beam_ids: Sequence[BeamId]) -> Dict:
        return {
            "scheme": self.scheme,
            "case_id": self.case_id or self.scheme,
            "feedback_scheme": self.feedback_scheme or self.scheme,
            "algorithm": self.algorithm or self.metadata.get("algorithm"),
            "objective_value": self.objective_value,
            "links": [l.to_dict(beam_ids) for l in self.links],
            "metadata": self.metadata,
        }


def schedule(reports: List[UEReport],
             beam_ids: Sequence[BeamId],
             cfg: Dict,
             tbar_mbps: Dict[int, float] | None = None,
             link_adapter=None,
             algorithm: str | None = None,
             progress_callback: Callable[[str], None] | None = None) -> ScheduleResult:
    domain_mode = normalize_domain_mode(cfg["scheduler"].get("domain_mode", "global"))
    if domain_mode == "per_site_joint":
        return schedule_grouped_domains(reports, beam_ids, cfg, tbar_mbps, link_adapter,
                                        domain_mode=domain_mode, algorithm=algorithm,
                                        group_key=lambda r: 0 if r.site_id is None else int(r.site_id),
                                        progress_callback=progress_callback)
    if domain_mode == "per_sector_independent":
        return schedule_grouped_domains(reports, beam_ids, cfg, tbar_mbps, link_adapter,
                                        domain_mode=domain_mode, algorithm=algorithm,
                                        group_key=lambda r: 0 if r.serving_cell is None else int(r.serving_cell),
                                        progress_callback=progress_callback)
    return _schedule_single_domain(reports, beam_ids, cfg, tbar_mbps, link_adapter,
                                   domain_id=None, domain_mode="global", algorithm=algorithm,
                                   progress_callback=progress_callback)


def _schedule_single_domain(reports: List[UEReport],
                            beam_ids: Sequence[BeamId],
                            cfg: Dict,
                            tbar_mbps: Dict[int, float] | None = None,
                            link_adapter=None,
                            domain_id: int | None = None,
                            domain_mode: str = "global",
                            algorithm: str | None = None,
                            progress_callback: Callable[[str], None] | None = None) -> ScheduleResult:
    alg = str(algorithm or cfg["scheduler"].get("algorithm", "exhaustive"))
    if alg == "exhaustive":
        return exhaustive_schedule(reports, beam_ids, cfg, tbar_mbps, link_adapter,
                                   domain_id=domain_id, domain_mode=domain_mode)
    if alg == "greedy":
        return greedy_schedule(reports, beam_ids, cfg, tbar_mbps, link_adapter,
                               domain_id=domain_id, domain_mode=domain_mode,
                               progress_callback=progress_callback)
    if alg == "adaptive_lambda_greedy":
        return greedy_schedule(reports, beam_ids, cfg, tbar_mbps, link_adapter,
                               domain_id=domain_id, domain_mode=domain_mode,
                               force_adaptive_lambda=True,
                               algorithm_name="adaptive_lambda_greedy",
                               progress_callback=progress_callback)
    if alg == "hard_conflict_greedy":
        return hard_conflict_greedy_schedule(
            reports, beam_ids, cfg, tbar_mbps, link_adapter,
            domain_id=domain_id, domain_mode=domain_mode,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unknown scheduler.algorithm={alg}")


def schedule_grouped_domains(reports: List[UEReport],
                             beam_ids: Sequence[BeamId],
                             cfg: Dict,
                             tbar_mbps: Dict[int, float] | None = None,
                             link_adapter=None,
                             domain_mode: str = "per_site_joint",
                             group_key=None,
                             algorithm: str | None = None,
                             progress_callback: Callable[[str], None] | None = None) -> ScheduleResult:
    scheme = reports[0].scheme if reports else "unknown"
    key_fn = group_key or (lambda r: 0)
    by_domain: Dict[int, List[UEReport]] = {}
    for r in reports:
        by_domain.setdefault(int(key_fn(r)), []).append(r)

    links: List[ScheduledLink] = []
    objective = 0.0
    domains: List[Dict] = []
    for domain_id in sorted(by_domain):
        res = _schedule_single_domain(by_domain[domain_id], beam_ids, cfg, tbar_mbps, link_adapter,
                                      domain_id=domain_id, domain_mode=domain_mode,
                                      algorithm=algorithm,
                                      progress_callback=progress_callback)
        links.extend(res.links)
        objective += float(res.objective_value)
        domains.append(res.metadata)

    metadata = {
        "domain_mode": domain_mode,
        "algorithm": str(algorithm or cfg["scheduler"].get("algorithm", "exhaustive")),
        "num_domains": len(domains),
        "domains": domains,
        "aggregate_stats": _aggregate_domain_stats(domains),
    }
    return ScheduleResult(scheme=scheme, objective_value=float(objective), links=links, metadata=metadata)


def schedule_per_site_joint(reports: List[UEReport],
                            beam_ids: Sequence[BeamId],
                            cfg: Dict,
                            tbar_mbps: Dict[int, float] | None = None,
                            link_adapter=None) -> ScheduleResult:
    return schedule_grouped_domains(
        reports, beam_ids, cfg, tbar_mbps, link_adapter,
        domain_mode="per_site_joint",
        group_key=lambda r: 0 if r.site_id is None else int(r.site_id),
    )


def _aggregate_domain_stats(domains: Sequence[Dict]) -> Dict:
    totals: Dict[str, int | float] = {}
    additive = {
        "num_reports_input",
        "num_reports_with_candidates",
        "num_reports_after_pruning",
        "raw_assignment_count",
        "assignment_count_after_zero_prune",
        "evaluated_assignment_count",
        "panel_pruned_count",
        "bound_pruned_count",
        "zero_upper_bound_pruned_reports",
        "num_scheduled",
        "num_su_outage_candidates",
        "num_scheduled_outage",
    }
    for d in domains:
        stats = d.get("stats", d)
        for k in additive:
            if k in stats:
                totals[k] = totals.get(k, 0) + int(stats.get(k, 0))
    return totals


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
                          link_adapter=None,
                          penalty_lambda: float | None = None) -> Tuple[float, List[ScheduledLink]]:
    if not assignments:
        return 0.0, []

    rep_by_ue = {r.ue_id: r for r in reports}
    scheme = reports[0].scheme if reports else "unknown"
    rate_kwargs = _rate_kwargs(cfg)
    if penalty_lambda is None:
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
                outage = bool(link_adapter.is_outage_from_sinr_lin(float(pred_sinr_lin), mcs))
            else:
                mcs = select_mcs_from_sinr_lin(pred_sinr_lin).index
                outage = bool(bler_from_sinr_db(float(lin_to_db(pred_sinr_lin)), mcs) > 0.1)
            pred_sinr_db = float(lin_to_db(pred_sinr_lin))
        else:
            # Limited-feedback scheduler uses SU MCS/rate and an ID-only conflict
            # penalty for proposed reports.
            mcs = int(cand.su_mcs)
            pred_sinr_db = float(cand.su_snr_db)
            outage = bool(cand.su_outage)
            for other_ue, other_b in assignments:
                if other_ue != ue_id and other_b in cand.conflict_beams:
                    conflict_penalty += 1.0

        if outage:
            r_mbps = 0.0
        elif link_adapter is not None:
            r_mbps = float(link_adapter.rate_mbps(mcs))
        else:
            r_mbps = rate_mbps_from_mcs(mcs, **rate_kwargs)
        utility += _pf_weight(ue_id, cfg, tbar_mbps) * r_mbps
        links.append(ScheduledLink(ue_id=ue_id,
                                   beam_index=b,
                                   predicted_sinr_db=pred_sinr_db,
                                   predicted_mcs=mcs,
                                   predicted_rate_mbps=r_mbps,
                                   predicted_outage=outage))
    if scheme in ("topk_conflict_id", "threshold_conflict_set"):
        utility -= penalty_lambda * conflict_penalty
    return float(utility), links


def _candidate_rate_mbps(rep: UEReport,
                         beam_index: int,
                         cfg: Dict,
                         link_adapter=None) -> float:
    cand = rep.candidate_by_beam(beam_index)
    if cand is None:
        return 0.0
    if cand.su_outage:
        return 0.0
    if link_adapter is not None:
        return float(link_adapter.rate_mbps(int(cand.su_mcs)))
    return float(rate_mbps_from_mcs(int(cand.su_mcs), **_rate_kwargs(cfg)))


def _resolve_conflict_penalty(reports: Sequence[UEReport],
                              cfg: Dict,
                              link_adapter=None,
                              force_adaptive: bool = False) -> Dict:
    scheduler_cfg = cfg["scheduler"]
    mode = str(scheduler_cfg.get("conflict_penalty_mode", "fixed")).lower()
    if force_adaptive:
        mode = "adaptive"
    rates = [
        _candidate_rate_mbps(r, c.beam_index, cfg, link_adapter)
        for r in reports
        for c in r.candidates
    ]
    median_rate = float(np.median(np.asarray(rates, dtype=float))) if rates else 0.0
    alpha = float(scheduler_cfg.get("adaptive_lambda_alpha", 0.2))
    if mode == "adaptive":
        value = alpha * median_rate
    elif mode == "fixed":
        value = float(scheduler_cfg.get("conflict_penalty_lambda", 0.0))
    else:
        raise ValueError(f"Unknown scheduler.conflict_penalty_mode={mode}")
    return {
        "conflict_penalty_mode": mode,
        "conflict_penalty_lambda_mbps": float(value),
        "adaptive_lambda_alpha": float(alpha),
        "candidate_su_rate_median_mbps": float(median_rate),
        "candidate_count_for_lambda": int(len(rates)),
    }


def _candidate_upper_bound(rep: UEReport,
                           beam_index: int,
                           cfg: Dict,
                           tbar_mbps: Dict[int, float] | None,
                           link_adapter=None) -> float:
    r_mbps = _candidate_rate_mbps(rep, beam_index, cfg, link_adapter)
    return float(_pf_weight(rep.ue_id, cfg, tbar_mbps) * r_mbps)


def _report_upper_bound(rep: UEReport,
                        cfg: Dict,
                        tbar_mbps: Dict[int, float] | None,
                        link_adapter=None) -> float:
    if not rep.candidates:
        return 0.0
    return max(_candidate_upper_bound(rep, c.beam_index, cfg, tbar_mbps, link_adapter)
               for c in rep.candidates)


def _sorted_report_copy(rep: UEReport,
                        cfg: Dict,
                        tbar_mbps: Dict[int, float] | None,
                        link_adapter=None) -> UEReport:
    cands = sorted(rep.candidates,
                   key=lambda c: _candidate_upper_bound(rep, c.beam_index, cfg, tbar_mbps, link_adapter),
                   reverse=True)
    return UEReport(ue_id=rep.ue_id,
                    scheme=rep.scheme,
                    candidates=cands,
                    site_id=rep.site_id,
                    serving_cell=rep.serving_cell,
                    full_gamma=rep.full_gamma,
                    full_service_power_w=rep.full_service_power_w,
                    full_noise_power_w=rep.full_noise_power_w)


def _count_candidate_assignments(reports: Sequence[UEReport], max_q: int) -> int:
    n = len(reports)
    qmax = min(int(max_q), n)
    total = 0
    counts = [len(r.candidates) for r in reports]
    for q in range(1, qmax + 1):
        for subset in combinations(range(n), q):
            prod = 1
            for i in subset:
                prod *= int(counts[i])
            total += prod
    return int(total)


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
                        link_adapter=None,
                        domain_id: int | None = None,
                        domain_mode: str = "global") -> ScheduleResult:
    max_q = _effective_max_mu_order(cfg)
    num_reports_input = len(reports)
    reports = [r for r in reports if r.candidates]
    num_reports_with_candidates = len(reports)
    num_su_outage_candidates = sum(int(c.su_outage) for r in reports for c in r.candidates)
    best_val = 0.0
    best_links: List[ScheduledLink] = []
    scheme = reports[0].scheme if reports else "unknown"
    pruning_cfg = cfg["scheduler"].get("exhaustive_pruning", {}) or {}
    sort_by_bound = bool(pruning_cfg.get("sort_by_upper_bound", True))
    zero_prune = bool(pruning_cfg.get("zero_upper_bound", True))
    branch_and_bound = bool(pruning_cfg.get("branch_and_bound", pruning_cfg.get("enabled", True)))
    use_panel_constraint = bool(cfg["scheduler"].get("use_panel_constraint", True))
    penalty_info = _resolve_conflict_penalty(reports, cfg, link_adapter)
    penalty_lambda = float(penalty_info["conflict_penalty_lambda_mbps"])

    raw_assignment_count = _count_candidate_assignments(reports, max_q)
    if sort_by_bound:
        reports = [_sorted_report_copy(r, cfg, tbar_mbps, link_adapter) for r in reports]

    report_bounds = [_report_upper_bound(r, cfg, tbar_mbps, link_adapter) for r in reports]
    zero_pruned = 0
    if zero_prune:
        kept: List[UEReport] = []
        kept_bounds: List[float] = []
        for r, ub in zip(reports, report_bounds):
            if ub > 0.0:
                kept.append(r)
                kept_bounds.append(float(ub))
            else:
                zero_pruned += 1
        reports = kept
        report_bounds = kept_bounds

    if sort_by_bound:
        order = np.argsort(np.asarray(report_bounds, dtype=float))[::-1]
        reports = [reports[int(i)] for i in order]
        report_bounds = [float(report_bounds[int(i)]) for i in order]

    assignment_count_after_zero = _count_candidate_assignments(reports, max_q)
    stats = {
        "algorithm": "exhaustive",
        "domain_id": None if domain_id is None else int(domain_id),
        "num_reports_input": int(num_reports_input),
        "num_reports_with_candidates": int(num_reports_with_candidates),
        "num_su_outage_candidates": int(num_su_outage_candidates),
        "num_reports_after_pruning": int(len(reports)),
        "max_mu_order": int(max_q),
        "raw_assignment_count": int(raw_assignment_count),
        "assignment_count_after_zero_prune": int(assignment_count_after_zero),
        "evaluated_assignment_count": 0,
        "panel_pruned_count": 0,
        "bound_pruned_count": 0,
        "zero_upper_bound_pruned_reports": int(zero_pruned),
        "branch_and_bound_enabled": branch_and_bound,
        "panel_constraint_enabled": use_panel_constraint,
        "candidate_ordering": "standalone_rate_desc" if sort_by_bound else "config_order",
        **penalty_info,
    }

    def remaining_upper_bound(start_index: int, slots: int) -> float:
        if slots <= 0 or start_index >= len(report_bounds):
            return 0.0
        return float(sum(report_bounds[start_index:start_index + int(slots)]))

    def dfs(start_index: int,
            assignments: List[Tuple[int, int]],
            used_panel_keys: set,
            current_val: float,
            current_links: List[ScheduledLink]) -> None:
        nonlocal best_val, best_links
        if assignments and current_val > best_val:
            best_val = float(current_val)
            best_links = current_links
        if len(assignments) >= min(max_q, len(reports)):
            return
        slots_left = int(max_q) - len(assignments)
        if branch_and_bound and current_val + remaining_upper_bound(start_index, slots_left) <= best_val + 1e-12:
            stats["bound_pruned_count"] += 1
            return
        for i in range(start_index, len(reports)):
            if branch_and_bound:
                suffix_bound = current_val + report_bounds[i] + remaining_upper_bound(i + 1, slots_left - 1)
                if suffix_bound <= best_val + 1e-12:
                    stats["bound_pruned_count"] += 1
                    continue
            r = reports[i]
            for c in r.candidates:
                key = beam_ids[c.beam_index].panel_key()
                if use_panel_constraint and key in used_panel_keys:
                    stats["panel_pruned_count"] += 1
                    continue
                trial = assignments + [(r.ue_id, c.beam_index)]
                val, links = _evaluate_assignments(
                    trial, reports, beam_ids, cfg, tbar_mbps, link_adapter,
                    penalty_lambda=penalty_lambda,
                )
                stats["evaluated_assignment_count"] += 1
                if not np.isfinite(val):
                    continue
                next_keys = set(used_panel_keys)
                if use_panel_constraint:
                    next_keys.add(key)
                dfs(i + 1, trial, next_keys, float(val), links)

    dfs(0, [], set(), 0.0, [])
    stats["best_objective_value"] = float(best_val)
    stats["num_scheduled"] = int(len(best_links))
    stats["num_scheduled_outage"] = int(sum(link.predicted_outage for link in best_links))
    metadata = {"domain_mode": domain_mode,
                "domain_id": None if domain_id is None else int(domain_id),
                "stats": stats}
    return ScheduleResult(scheme=scheme, objective_value=float(best_val), links=best_links, metadata=metadata)


def _map_scheduler_sinr_lin(sinr_lin: np.ndarray,
                            cfg: Dict,
                            link_adapter=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(sinr_lin, dtype=float)
    if link_adapter is not None and hasattr(link_adapter, "map_sinr_lin"):
        mcs, outage, rates = link_adapter.map_sinr_lin(values)
        return (np.asarray(mcs, dtype=np.int32), np.asarray(outage, dtype=bool),
                np.asarray(rates, dtype=float))

    flat = values.reshape(-1)
    mcs = np.empty(flat.size, dtype=np.int32)
    outage = np.empty(flat.size, dtype=bool)
    rates = np.empty(flat.size, dtype=float)
    rate_kwargs = _rate_kwargs(cfg)
    for i, value in enumerate(flat):
        if link_adapter is not None:
            selected = int(link_adapter.select_mcs_from_sinr_lin(float(value)))
            is_outage = bool(link_adapter.is_outage_from_sinr_lin(float(value), selected))
            rate = 0.0 if is_outage else float(link_adapter.rate_mbps(selected))
        else:
            selected = int(select_mcs_from_sinr_lin(float(value)).index)
            is_outage = bool(bler_from_sinr_db(float(lin_to_db(value)), selected) > 0.1)
            rate = 0.0 if is_outage else float(rate_mbps_from_mcs(selected, **rate_kwargs))
        mcs[i] = selected
        outage[i] = is_outage
        rates[i] = rate
    return mcs.reshape(values.shape), outage.reshape(values.shape), rates.reshape(values.shape)


def _node_arrays(reports: Sequence[UEReport],
                 beam_ids: Sequence[BeamId],
                 cfg: Dict,
                 tbar_mbps: Dict[int, float] | None,
                 link_adapter=None):
    nodes = [(r, c) for r in reports for c in r.candidates]
    ue_ids = np.asarray([r.ue_id for r, _ in nodes], dtype=np.int32)
    beam_indices = np.asarray([c.beam_index for _, c in nodes], dtype=np.int32)
    panel_keys = [beam_ids[int(b)].panel_key() for b in beam_indices]
    weights = np.asarray([_pf_weight(r.ue_id, cfg, tbar_mbps) for r, _ in nodes], dtype=float)
    rates = np.asarray([
        0.0 if c.su_outage else (
            float(link_adapter.rate_mbps(int(c.su_mcs))) if link_adapter is not None
            else float(rate_mbps_from_mcs(int(c.su_mcs), **_rate_kwargs(cfg)))
        )
        for _, c in nodes
    ], dtype=float)
    return nodes, ue_ids, beam_indices, panel_keys, weights, rates


def _round_progress(progress_callback: Callable[[str], None] | None,
                    domain_id: int | None,
                    q: int,
                    max_q: int,
                    candidates: int,
                    elapsed_s: float) -> None:
    if progress_callback is None:
        return
    domain = "" if domain_id is None else f"domain={domain_id}, "
    progress_callback(f"{domain}q={q}/{max_q}, candidates={candidates}, elapsed={elapsed_s:.3f}s")


def _optimized_limited_greedy_schedule(reports: List[UEReport],
                                        beam_ids: Sequence[BeamId],
                                        cfg: Dict,
                                        tbar_mbps: Dict[int, float] | None,
                                        link_adapter,
                                        domain_id: int | None,
                                        domain_mode: str,
                                        force_adaptive_lambda: bool,
                                        algorithm_name: str,
                                        progress_callback: Callable[[str], None] | None) -> ScheduleResult:
    started = perf_counter()
    max_q = _effective_max_mu_order(cfg)
    num_reports_input = len(reports)
    reports = [r for r in reports if r.candidates]
    scheme = reports[0].scheme if reports else "unknown"
    nodes, ue_ids, beam_indices, panel_keys, weights, rates = _node_arrays(
        reports, beam_ids, cfg, tbar_mbps, link_adapter,
    )
    count = len(nodes)
    penalty_info = _resolve_conflict_penalty(
        reports, cfg, link_adapter, force_adaptive=force_adaptive_lambda,
    )
    penalty_lambda = float(penalty_info["conflict_penalty_lambda_mbps"])
    adjacency = np.zeros((count, count), dtype=np.uint8)
    if scheme in ("topk_conflict_id", "threshold_conflict_set"):
        for i, (_, cand) in enumerate(nodes):
            if not cand.conflict_beams:
                continue
            for j in range(count):
                if ue_ids[j] != ue_ids[i] and int(beam_indices[j]) in cand.conflict_beams:
                    adjacency[i, j] = 1

    base_utility = weights * rates
    conflict_increment = np.zeros(count, dtype=float)
    selected: List[int] = []
    used_ues: set[int] = set()
    used_panel_keys: set = set()
    current_val = 0.0
    use_panel_constraint = bool(cfg["scheduler"].get("use_panel_constraint", True))
    rounds: List[Dict] = []
    evaluated_count = 0
    panel_pruned_count = 0

    while len(selected) < max_q:
        q = len(selected) + 1
        remaining = np.asarray([int(u) not in used_ues for u in ue_ids], dtype=bool)
        remaining_count = int(np.count_nonzero(remaining))
        if use_panel_constraint:
            panel_legal = np.asarray([key not in used_panel_keys for key in panel_keys], dtype=bool)
            pruned = int(np.count_nonzero(remaining & ~panel_legal))
        else:
            panel_legal = np.ones(count, dtype=bool)
            pruned = 0
        eligible = remaining & panel_legal
        eligible_indices = np.flatnonzero(eligible)
        candidate_count = int(eligible_indices.size)
        panel_pruned_count += pruned
        evaluated_count += candidate_count
        chosen = None
        best_delta = 0.0
        if candidate_count:
            delta = base_utility[eligible_indices] - penalty_lambda * conflict_increment[eligible_indices]
            local = int(np.argmax(delta))
            if float(delta[local]) > 0.0:
                chosen = int(eligible_indices[local])
                best_delta = float(delta[local])
        elapsed = float(perf_counter() - started)
        rounds.append({
            "q": int(q),
            "remaining_candidate_count": remaining_count,
            "evaluated_candidate_count": candidate_count,
            "panel_pruned_count": pruned,
            "elapsed_s": elapsed,
        })
        _round_progress(progress_callback, domain_id, q, max_q, candidate_count, elapsed)
        if chosen is None:
            break
        selected.append(chosen)
        used_ues.add(int(ue_ids[chosen]))
        if use_panel_constraint:
            used_panel_keys.add(panel_keys[chosen])
        current_val += best_delta
        if count:
            conflict_increment += adjacency[:, chosen].astype(float) + adjacency[chosen, :].astype(float)

    assignments = [(int(ue_ids[i]), int(beam_indices[i])) for i in selected]
    final_val, final_links = _evaluate_assignments(
        assignments, reports, beam_ids, cfg, tbar_mbps, link_adapter,
        penalty_lambda=penalty_lambda,
    )
    stats = {
        "algorithm": algorithm_name,
        "implementation": "v2.10_incremental_limited_feedback",
        "domain_id": None if domain_id is None else int(domain_id),
        "num_reports_input": int(num_reports_input),
        "num_reports_with_candidates": int(len(reports)),
        "num_su_outage_candidates": int(sum(int(c.su_outage) for r in reports for c in r.candidates)),
        "max_mu_order": int(max_q),
        "evaluated_assignment_count": int(evaluated_count),
        "panel_pruned_count": int(panel_pruned_count),
        "candidate_node_count": int(count),
        "directed_conflict_edge_count": int(np.sum(adjacency)),
        "rounds": rounds,
        "scheduler_domain_elapsed_s": float(perf_counter() - started),
        **penalty_info,
        "best_objective_value": float(final_val),
        "num_scheduled": int(len(final_links)),
        "num_scheduled_outage": int(sum(link.predicted_outage for link in final_links)),
    }
    metadata = {"domain_mode": domain_mode,
                "domain_id": None if domain_id is None else int(domain_id),
                "stats": stats}
    return ScheduleResult(scheme=scheme, objective_value=float(final_val), links=final_links, metadata=metadata)


def _optimized_full_gamma_greedy_schedule(reports: List[UEReport],
                                           beam_ids: Sequence[BeamId],
                                           cfg: Dict,
                                           tbar_mbps: Dict[int, float] | None,
                                           link_adapter,
                                           domain_id: int | None,
                                           domain_mode: str,
                                           algorithm_name: str,
                                           progress_callback: Callable[[str], None] | None) -> ScheduleResult:
    started = perf_counter()
    max_q = _effective_max_mu_order(cfg)
    num_reports_input = len(reports)
    reports = [r for r in reports if r.candidates]
    scheme = reports[0].scheme if reports else "unknown"
    nodes, ue_ids, beam_indices, panel_keys, weights, _ = _node_arrays(
        reports, beam_ids, cfg, tbar_mbps, link_adapter,
    )
    count = len(nodes)
    signal = np.asarray([
        float(r.full_service_power_w[int(c.beam_index)]) for r, c in nodes
    ], dtype=float) if count else np.asarray([], dtype=float)
    noise = np.asarray([
        float(r.full_noise_power_w) for r, _ in nodes
    ], dtype=float) if count else np.asarray([], dtype=float)
    interference = np.zeros((count, count), dtype=float)
    for i, (rep, cand) in enumerate(nodes):
        s = signal[i]
        n = noise[i]
        for j in range(count):
            g = float(rep.full_gamma[int(cand.beam_index), int(beam_indices[j])])
            interference[i, j] = max(0.0, s / max(g, 1e-30) - n)

    selected: List[int] = []
    selected_den = np.asarray([], dtype=float)
    incoming_den = noise.copy()
    used_ues: set[int] = set()
    used_panel_keys: set = set()
    current_val = 0.0
    use_panel_constraint = bool(cfg["scheduler"].get("use_panel_constraint", True))
    rounds: List[Dict] = []
    evaluated_count = 0
    panel_pruned_count = 0

    while len(selected) < max_q:
        q = len(selected) + 1
        remaining = np.asarray([int(u) not in used_ues for u in ue_ids], dtype=bool)
        remaining_count = int(np.count_nonzero(remaining))
        if use_panel_constraint:
            panel_legal = np.asarray([key not in used_panel_keys for key in panel_keys], dtype=bool)
            pruned = int(np.count_nonzero(remaining & ~panel_legal))
        else:
            panel_legal = np.ones(count, dtype=bool)
            pruned = 0
        eligible_indices = np.flatnonzero(remaining & panel_legal)
        candidate_count = int(eligible_indices.size)
        panel_pruned_count += pruned
        evaluated_count += candidate_count
        chosen = None
        best_val = current_val
        best_delta = 0.0
        if candidate_count:
            new_sinr = signal[eligible_indices] / np.maximum(incoming_den[eligible_indices], 1e-30)
            if selected:
                selected_idx = np.asarray(selected, dtype=int)
                existing_den = selected_den[:, None] + interference[np.ix_(selected_idx, eligible_indices)]
                existing_sinr = signal[selected_idx, None] / np.maximum(existing_den, 1e-30)
                trial_sinr = np.vstack((existing_sinr, new_sinr[None, :]))
                _, _, trial_rates = _map_scheduler_sinr_lin(trial_sinr, cfg, link_adapter)
                trial_utility = (
                    np.sum(weights[selected_idx, None] * trial_rates[:-1, :], axis=0)
                    + weights[eligible_indices] * trial_rates[-1, :]
                )
            else:
                _, _, trial_rates = _map_scheduler_sinr_lin(new_sinr, cfg, link_adapter)
                trial_utility = weights[eligible_indices] * trial_rates
            delta = trial_utility - current_val
            local = int(np.argmax(delta))
            if float(delta[local]) > 0.0:
                chosen = int(eligible_indices[local])
                best_delta = float(delta[local])
                best_val = float(trial_utility[local])
        elapsed = float(perf_counter() - started)
        rounds.append({
            "q": int(q),
            "remaining_candidate_count": remaining_count,
            "evaluated_candidate_count": candidate_count,
            "panel_pruned_count": pruned,
            "elapsed_s": elapsed,
        })
        _round_progress(progress_callback, domain_id, q, max_q, candidate_count, elapsed)
        if chosen is None:
            break

        if selected:
            selected_idx = np.asarray(selected, dtype=int)
            selected_den = selected_den + interference[selected_idx, chosen]
        selected_den = np.append(selected_den, incoming_den[chosen])
        incoming_den = incoming_den + interference[:, chosen]
        selected.append(chosen)
        used_ues.add(int(ue_ids[chosen]))
        if use_panel_constraint:
            used_panel_keys.add(panel_keys[chosen])
        current_val = best_val

    assignments = [(int(ue_ids[i]), int(beam_indices[i])) for i in selected]
    final_val, final_links = _evaluate_assignments(
        assignments, reports, beam_ids, cfg, tbar_mbps, link_adapter,
    )
    stats = {
        "algorithm": algorithm_name,
        "implementation": "v2.10_incremental_vectorized_full_gamma",
        "domain_id": None if domain_id is None else int(domain_id),
        "num_reports_input": int(num_reports_input),
        "num_reports_with_candidates": int(len(reports)),
        "num_su_outage_candidates": int(sum(int(c.su_outage) for r in reports for c in r.candidates)),
        "max_mu_order": int(max_q),
        "evaluated_assignment_count": int(evaluated_count),
        "panel_pruned_count": int(panel_pruned_count),
        "candidate_node_count": int(count),
        "interference_matrix_elements": int(interference.size),
        "rounds": rounds,
        "scheduler_domain_elapsed_s": float(perf_counter() - started),
        "best_objective_value": float(final_val),
        "num_scheduled": int(len(final_links)),
        "num_scheduled_outage": int(sum(link.predicted_outage for link in final_links)),
    }
    metadata = {"domain_mode": domain_mode,
                "domain_id": None if domain_id is None else int(domain_id),
                "stats": stats}
    return ScheduleResult(scheme=scheme, objective_value=float(final_val), links=final_links, metadata=metadata)


def greedy_schedule(reports: List[UEReport],
                    beam_ids: Sequence[BeamId],
                    cfg: Dict,
                    tbar_mbps: Dict[int, float] | None = None,
                    link_adapter=None,
                    domain_id: int | None = None,
                    domain_mode: str = "global",
                    force_adaptive_lambda: bool = False,
                    algorithm_name: str = "greedy",
                    progress_callback: Callable[[str], None] | None = None) -> ScheduleResult:
    if not bool(cfg["scheduler"].get("optimized_greedy", True)):
        return _legacy_greedy_schedule(
            reports, beam_ids, cfg, tbar_mbps, link_adapter, domain_id, domain_mode,
            force_adaptive_lambda, algorithm_name, progress_callback,
        )
    active_reports = [r for r in reports if r.candidates]
    scheme = active_reports[0].scheme if active_reports else "unknown"
    if scheme == "full_gamma":
        return _optimized_full_gamma_greedy_schedule(
            reports, beam_ids, cfg, tbar_mbps, link_adapter, domain_id, domain_mode,
            algorithm_name, progress_callback,
        )
    return _optimized_limited_greedy_schedule(
        reports, beam_ids, cfg, tbar_mbps, link_adapter, domain_id, domain_mode,
        force_adaptive_lambda, algorithm_name, progress_callback,
    )


def _legacy_greedy_schedule(reports: List[UEReport],
                            beam_ids: Sequence[BeamId],
                            cfg: Dict,
                            tbar_mbps: Dict[int, float] | None = None,
                            link_adapter=None,
                            domain_id: int | None = None,
                            domain_mode: str = "global",
                            force_adaptive_lambda: bool = False,
                            algorithm_name: str = "greedy",
                            progress_callback: Callable[[str], None] | None = None) -> ScheduleResult:
    started = perf_counter()
    max_q = _effective_max_mu_order(cfg)
    num_reports_input = len(reports)
    reports = [r for r in reports if r.candidates]
    scheme = reports[0].scheme if reports else "unknown"
    num_su_outage_candidates = sum(int(c.su_outage) for r in reports for c in r.candidates)
    current: List[Tuple[int, int]] = []
    current_val = 0.0
    used_ues = set()
    penalty_info = _resolve_conflict_penalty(
        reports, cfg, link_adapter, force_adaptive=force_adaptive_lambda,
    )
    penalty_lambda = float(penalty_info["conflict_penalty_lambda_mbps"])
    stats = {
        "algorithm": algorithm_name,
        "implementation": "v2.9_reference_greedy",
        "domain_id": None if domain_id is None else int(domain_id),
        "num_reports_input": int(num_reports_input),
        "num_reports_with_candidates": int(len(reports)),
        "num_su_outage_candidates": int(num_su_outage_candidates),
        "max_mu_order": int(max_q),
        "evaluated_assignment_count": 0,
        "panel_pruned_count": 0,
        **penalty_info,
    }
    rounds: List[Dict] = []

    while len(current) < max_q:
        q = len(current) + 1
        evaluated_before = int(stats["evaluated_assignment_count"])
        pruned_before = int(stats["panel_pruned_count"])
        remaining_count = sum(len(r.candidates) for r in reports if r.ue_id not in used_ues)
        best_delta = 0.0
        best_assignment = None
        best_val = current_val
        for r in reports:
            if r.ue_id in used_ues:
                continue
            for c in r.candidates:
                trial = current + [(r.ue_id, c.beam_index)]
                if not _panel_constraint_ok(trial, reports, beam_ids, cfg):
                    stats["panel_pruned_count"] += 1
                    continue
                val, _ = _evaluate_assignments(
                    trial, reports, beam_ids, cfg, tbar_mbps, link_adapter,
                    penalty_lambda=penalty_lambda,
                )
                stats["evaluated_assignment_count"] += 1
                delta = val - current_val
                if delta > best_delta:
                    best_delta = delta
                    best_assignment = (r.ue_id, c.beam_index)
                    best_val = val
        candidate_count = int(stats["evaluated_assignment_count"]) - evaluated_before
        elapsed = float(perf_counter() - started)
        rounds.append({
            "q": int(q),
            "remaining_candidate_count": int(remaining_count),
            "evaluated_candidate_count": int(candidate_count),
            "panel_pruned_count": int(stats["panel_pruned_count"]) - pruned_before,
            "elapsed_s": elapsed,
        })
        _round_progress(progress_callback, domain_id, q, max_q, candidate_count, elapsed)
        if best_assignment is None:
            break
        current.append(best_assignment)
        used_ues.add(best_assignment[0])
        current_val = best_val
    final_val, final_links = _evaluate_assignments(
        current, reports, beam_ids, cfg, tbar_mbps, link_adapter,
        penalty_lambda=penalty_lambda,
    )
    stats["best_objective_value"] = float(final_val)
    stats["num_scheduled"] = int(len(final_links))
    stats["num_scheduled_outage"] = int(sum(link.predicted_outage for link in final_links))
    stats["rounds"] = rounds
    stats["scheduler_domain_elapsed_s"] = float(perf_counter() - started)
    metadata = {"domain_mode": domain_mode,
                "domain_id": None if domain_id is None else int(domain_id),
                "stats": stats}
    return ScheduleResult(scheme=scheme, objective_value=final_val, links=final_links, metadata=metadata)


def hard_conflict_greedy_schedule(reports: List[UEReport],
                                  beam_ids: Sequence[BeamId],
                                  cfg: Dict,
                                  tbar_mbps: Dict[int, float] | None = None,
                                  link_adapter=None,
                                  domain_id: int | None = None,
                                  domain_mode: str = "global",
                                  progress_callback: Callable[[str], None] | None = None) -> ScheduleResult:
    """Greedy maximum-weight independent set over reported (UE, beam) nodes.

    Selecting one node removes the selected UE's other nodes and only the
    conflicting candidate nodes of other UEs. It never removes an entire UE
    merely because one of that UE's candidate beams conflicts.
    """
    started = perf_counter()
    max_q = _effective_max_mu_order(cfg)
    num_reports_input = len(reports)
    reports = [r for r in reports if r.candidates]
    scheme = reports[0].scheme if reports else "unknown"
    num_su_outage_candidates = sum(int(c.su_outage) for r in reports for c in r.candidates)
    rep_by_ue = {r.ue_id: r for r in reports}
    pool: Dict[Tuple[int, int], float] = {
        (r.ue_id, c.beam_index): _candidate_rate_mbps(r, c.beam_index, cfg, link_adapter)
        for r in reports
        for c in r.candidates
        if not c.su_outage
    }
    initial_pool_size = len(pool)
    selected: List[Tuple[int, int]] = []
    removed_same_ue = 0
    removed_conflict = 0
    removed_panel = 0
    use_panel_constraint = bool(cfg["scheduler"].get("use_panel_constraint", True))
    rounds: List[Dict] = []

    def conflicts(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        ua, ba = a
        ub, bb = b
        ca = rep_by_ue[ua].candidate_by_beam(ba)
        cb = rep_by_ue[ub].candidate_by_beam(bb)
        return bool(
            (ca is not None and bb in ca.conflict_beams)
            or (cb is not None and ba in cb.conflict_beams)
        )

    while pool and len(selected) < max_q:
        q = len(selected) + 1
        candidate_count = len(pool)
        removed_panel_before = removed_panel
        # The tuple suffix gives deterministic tie-breaking across runs.
        chosen = min(pool, key=lambda node: (-pool[node], node[0], node[1]))
        selected.append(chosen)
        chosen_ue, chosen_beam = chosen
        chosen_panel = beam_ids[chosen_beam].panel_key()
        del pool[chosen]

        for node in list(pool):
            if node[0] == chosen_ue:
                del pool[node]
                removed_same_ue += 1
            elif conflicts(chosen, node):
                del pool[node]
                removed_conflict += 1
            elif use_panel_constraint and beam_ids[node[1]].panel_key() == chosen_panel:
                del pool[node]
                removed_panel += 1
        elapsed = float(perf_counter() - started)
        rounds.append({
            "q": int(q),
            "remaining_candidate_count": int(candidate_count),
            "evaluated_candidate_count": int(candidate_count),
            "panel_pruned_count": int(removed_panel - removed_panel_before),
            "elapsed_s": elapsed,
        })
        _round_progress(progress_callback, domain_id, q, max_q, candidate_count, elapsed)

    final_val, final_links = _evaluate_assignments(
        selected, reports, beam_ids, cfg, tbar_mbps, link_adapter, penalty_lambda=0.0,
    )
    stats = {
        "algorithm": "hard_conflict_greedy",
        "implementation": "v2.10_candidate_pool_hard_conflict",
        "domain_id": None if domain_id is None else int(domain_id),
        "num_reports_input": int(num_reports_input),
        "num_reports_with_candidates": int(len(reports)),
        "num_su_outage_candidates": int(num_su_outage_candidates),
        "max_mu_order": int(max_q),
        "initial_candidate_pool_size": int(initial_pool_size),
        "removed_same_ue_candidates": int(removed_same_ue),
        "removed_conflicting_candidates": int(removed_conflict),
        "removed_panel_candidates": int(removed_panel),
        "panel_constraint_enabled": use_panel_constraint,
        "node_weight": "su_rate_mbps",
        "best_objective_value": float(final_val),
        "num_scheduled": int(len(final_links)),
        "num_scheduled_outage": int(sum(link.predicted_outage for link in final_links)),
        "rounds": rounds,
        "scheduler_domain_elapsed_s": float(perf_counter() - started),
    }
    metadata = {
        "domain_mode": domain_mode,
        "domain_id": None if domain_id is None else int(domain_id),
        "stats": stats,
    }
    return ScheduleResult(
        scheme=scheme,
        objective_value=float(final_val),
        links=final_links,
        metadata=metadata,
    )
