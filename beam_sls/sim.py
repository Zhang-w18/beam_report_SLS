from __future__ import annotations

import copy
import json
import os
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np

from .channel import SionnaImportProbe, generate_channel
from .codebook import ArrayConfig, build_network_tx_beams, dft_codebook_from_array
from .config import save_config
from .coverage import compute_coverage_heatmap_standard_sampling, compute_fixed_vertical_beam_cdf
from .evaluation import EvaluationCase, resolve_evaluation_plan
from .feedback import make_reports
from .link import LinkEvalRow, run_tti_loop
from .link_adaptation import make_link_adapter, make_scheduler_link_adapter
from .measurement import compute_gamma_measurement
from .rf import resolve_rf_architecture, resolved_max_mu_order, tx_units_per_sector
from .plotting import plot_bar, plot_best_beam_heatmap, plot_cdf, plot_heatmap, plot_topology
from .scheduler import ScheduleResult, is_sector_domain_mode, is_site_domain_mode, normalize_domain_mode, schedule
from .topology import make_topology, topology_to_rows
from .utils import (dbm_to_watt, ensure_dir, occupied_bandwidth_hz, percentile,
                    thermal_noise_watt, watt_to_dbm, write_csv, write_json)


BASELINE_NO_INTERFERENCE_SCHEME = "baseline_no_interference_upper_bound"


def _asdict_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if hasattr(r, "__dict__"):
            out.append(dict(r.__dict__))
        else:
            out.append(dict(r))
    return out


def _panels_per_cell(cfg: Dict[str, Any], tx_cfg: ArrayConfig | None = None, rf_architecture=None) -> int:
    """Return independently schedulable TX units per sector/cell.

    The name is kept for backward compatibility with v2.x files. In v2.4 the
    value means TX units, not necessarily physical panels. It is resolved from
    the RF architecture: panel-polarization subarrays or fully-connected TXRUs.
    """
    if tx_cfg is not None:
        rf = rf_architecture or resolve_rf_architecture(cfg, tx_cfg)
        return int(tx_units_per_sector(cfg, rf))
    trp = cfg.get("trp", {})
    return int(trp.get("num_trps_per_sector", trp.get("num_panels_per_sector", 1)))


def _progress(cfg: Dict[str, Any], msg: str) -> None:
    if bool(cfg.get("progress", {}).get("enabled", True)):
        print(msg, flush=True)


def _beam_indices_by_site(beam_ids) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for bi, bid in enumerate(beam_ids):
        out.setdefault(int(bid.trp), []).append(int(bi))
    return out


def _beam_indices_by_cell(beam_ids) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for bi, bid in enumerate(beam_ids):
        out.setdefault(int(bid.cell), []).append(int(bi))
    return out


def _domain_candidate_beam_indices_by_ue(topo, beam_ids, domain_mode: str) -> Dict[int, List[int]] | None:
    if is_site_domain_mode(domain_mode):
        beams_by_site = _beam_indices_by_site(beam_ids)
        return {
            int(ue.ue_id): beams_by_site.get(int(ue.site_id), [])
            for ue in topo.ues
        }
    if is_sector_domain_mode(domain_mode):
        beams_by_cell = _beam_indices_by_cell(beam_ids)
        return {
            int(ue.ue_id): beams_by_cell.get(int(ue.serving_cell), [])
            for ue in topo.ues
        }
    return None


def _domain_metric(values: np.ndarray,
                   candidate_beam_indices_by_ue: Dict[int, List[int]] | None,
                   fn) -> float:
    if candidate_beam_indices_by_ue is None:
        arr = np.asarray(values, dtype=float).reshape(-1)
    else:
        chunks = []
        for u, inds in candidate_beam_indices_by_ue.items():
            if inds:
                chunks.append(np.asarray(values[int(u), inds], dtype=float))
        arr = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(fn(arr))


def _avg_domain_beams(candidate_beam_indices_by_ue: Dict[int, List[int]] | None,
                      num_beams: int,
                      num_ues: int) -> float:
    if num_ues <= 0:
        return 0.0
    if candidate_beam_indices_by_ue is None:
        return float(num_beams)
    return float(np.mean([len(v) for v in candidate_beam_indices_by_ue.values()]))


def _schedule_stat_rows(drop: int, scheme: str, sched) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta = getattr(sched, "metadata", {}) or {}
    domains = meta.get("domains")
    if domains:
        for d in domains:
            stats = dict(d.get("stats", {}))
            stats.pop("rounds", None)
            row = {
                "drop": int(drop),
                "scheme": scheme,
                "case_id": sched.case_id or scheme,
                "feedback_scheme": sched.feedback_scheme or sched.scheme,
                "algorithm": sched.algorithm or meta.get("algorithm"),
                "domain_mode": meta.get("domain_mode", "per_site_joint"),
                "domain_id": d.get("domain_id", stats.get("domain_id")),
            }
            row.update(stats)
            rows.append(row)
        agg = dict(meta.get("aggregate_stats", {}))
        if agg:
            row = {
                "drop": int(drop),
                "scheme": scheme,
                "case_id": sched.case_id or scheme,
                "feedback_scheme": sched.feedback_scheme or sched.scheme,
                "algorithm": sched.algorithm or meta.get("algorithm"),
                "domain_mode": meta.get("domain_mode", "per_site_joint"),
                "domain_id": "all",
            }
            row.update(agg)
            rows.append(row)
        return rows
    stats = dict(meta.get("stats", {}))
    stats.pop("rounds", None)
    row = {
        "drop": int(drop),
        "scheme": scheme,
        "case_id": sched.case_id or scheme,
        "feedback_scheme": sched.feedback_scheme or sched.scheme,
        "algorithm": sched.algorithm or meta.get("algorithm"),
        "domain_mode": meta.get("domain_mode", "global"),
        "domain_id": meta.get("domain_id", stats.get("domain_id")),
    }
    row.update(stats)
    rows.append(row)
    return rows


def _counter_snapshot(adapter) -> Dict[str, int]:
    if adapter is None or not hasattr(adapter, "snapshot_counters"):
        return {}
    return {k: int(v) for k, v in adapter.snapshot_counters().items()}


def _counter_delta(before: Dict[str, int], after: Dict[str, int], prefix: str = "") -> Dict[str, int]:
    keys = set(before) | set(after)
    return {f"{prefix}{key}": int(after.get(key, 0) - before.get(key, 0)) for key in sorted(keys)}


def _attach_case_scheduler_metrics(sched, metrics: Dict[str, Any]) -> None:
    meta = getattr(sched, "metadata", {}) or {}
    if meta.get("domains"):
        meta.setdefault("aggregate_stats", {}).update(metrics)
    else:
        meta.setdefault("stats", {}).update(metrics)


def _schedule_iteration_rows(drop: int, case_id: str, sched) -> List[Dict[str, Any]]:
    meta = getattr(sched, "metadata", {}) or {}
    domains = meta.get("domains") or [meta]
    rows: List[Dict[str, Any]] = []
    for domain in domains:
        stats = domain.get("stats", {})
        domain_id = domain.get("domain_id", stats.get("domain_id"))
        for round_row in stats.get("rounds", []) or []:
            rows.append({
                "drop": int(drop),
                "case_id": case_id,
                "feedback_scheme": sched.feedback_scheme or sched.scheme,
                "algorithm": sched.algorithm or meta.get("algorithm"),
                "domain_mode": meta.get("domain_mode", "global"),
                "domain_id": domain_id,
                **dict(round_row),
            })
    return rows


def _max_beams_from_cfg(array_section: Dict[str, Any], array_cfg: ArrayConfig) -> int | None:
    val = array_section.get("max_beams", None)
    if val is None:
        if array_cfg.num_beams_h is not None and array_cfg.num_beams_v is not None:
            return int(array_cfg.num_beams_h) * int(array_cfg.num_beams_v)
        return None
    return int(val)


def build_ue_goodput_rows(link_rows: List[Dict[str, Any]],
                          schemes: List[str],
                          ue_rows: List[Dict[str, Any]],
                          num_tti: int) -> List[Dict[str, Any]]:
    """Average every (drop, UE) over all TTIs, including unscheduled zeros."""
    sums: Dict[Tuple[str, int, int], float] = {}
    for r in link_rows:
        key = (str(r["scheme"]), int(r["drop"]), int(r["ue_id"]))
        sums[key] = sums.get(key, 0.0) + float(r["goodput_mbps"])
    ue_keys = sorted({(int(r["drop"]), int(r["ue_id"])) for r in ue_rows})
    divisor = max(1, int(num_tti))
    return [
        {
            "scheme": scheme,
            "drop": drop,
            "ue_id": ue_id,
            "avg_goodput_mbps": float(sums.get((scheme, drop, ue_id), 0.0) / divisor),
            "scheduled": int((scheme, drop, ue_id) in sums),
        }
        for scheme in schemes
        for drop, ue_id in ue_keys
    ]


def build_scheduled_ue_su_throughput_rows(drop: int,
                                          case_id: str,
                                          sched,
                                          reports,
                                          beam_ids,
                                          link_adapter) -> List[Dict[str, Any]]:
    """Build standalone-throughput samples for UEs selected by a schedule.

    Each sample uses the selected service beam's standalone SNR/MCS before
    adding interference from any co-scheduled UE.  It is intentionally
    different from predicted MU rate and realized TTI goodput.
    """
    reports_by_ue = {int(report.ue_id): report for report in reports}
    rows: List[Dict[str, Any]] = []
    for link in sched.links:
        report = reports_by_ue.get(int(link.ue_id))
        candidate = None if report is None else report.candidate_by_beam(int(link.beam_index))
        if candidate is None:
            raise ValueError(
                f"Scheduled UE {link.ue_id} beam {link.beam_index} is absent from its report"
            )
        su_outage = bool(candidate.su_outage)
        rows.append({
            "drop": int(drop),
            "scheme": case_id,
            "case_id": case_id,
            "feedback_scheme": sched.feedback_scheme or sched.scheme,
            "algorithm": sched.algorithm or sched.metadata.get("algorithm"),
            "ue_id": int(link.ue_id),
            "beam_index": int(link.beam_index),
            "beam_id": beam_ids[int(link.beam_index)].short(),
            "su_snr_db": float(candidate.su_snr_db),
            "su_mcs": int(candidate.su_mcs),
            "su_outage": su_outage,
            "su_throughput_mbps": (
                0.0 if su_outage else float(link_adapter.rate_mbps(int(candidate.su_mcs)))
            ),
        })
    return rows


def summarize_scheduled_ue_su_throughput(
        rows: List[Dict[str, Any]],
        schemes: List[str]) -> List[Dict[str, Any]]:
    """Summarize scheduled-UE SU-throughput samples."""
    summary: List[Dict[str, Any]] = []
    for scheme in schemes:
        scheme_rows = [row for row in rows if row.get("scheme") == scheme]
        values = [float(row["su_throughput_mbps"]) for row in scheme_rows]
        summary.append({
            "scheme": scheme,
            "num_scheduled_ue_samples": int(len(values)),
            "avg_su_throughput_mbps": float(np.mean(values)) if values else 0.0,
            "p05_su_throughput_mbps": percentile(values, 5.0),
            "p50_su_throughput_mbps": percentile(values, 50.0),
            "p95_su_throughput_mbps": percentile(values, 95.0),
            "su_outage_ratio": (
                float(np.mean([bool(row["su_outage"]) for row in scheme_rows]))
                if scheme_rows else 0.0
            ),
        })
    return summary


def build_paired_case_debug_lines(
        scheduled_pair_lists: Dict[Tuple[int, str], List[Tuple[int, int]]],
        link_rows: List[Dict[str, Any]],
        pairs: List[List[str]]) -> List[str]:
    """Return compact diagnostics for paired cases that should be identical."""
    lines = ["[paired-debug] begin"]
    float_fields = (
        "effective_sinr_db",
        "olla_offset_db",
        "mcs_selection_sinr_db",
        "tbler",
        "ack_random_uniform",
    )
    exact_fields = ("beam_index", "link_position", "actual_mcs", "ack", "goodput_bits")
    comparison_fields = (
        "beam_index",
        "link_position",
        "effective_sinr_db",
        "olla_offset_db",
        "mcs_selection_sinr_db",
        "actual_mcs",
        "tbler",
        "ack_random_uniform",
        "ack",
        "goodput_bits",
    )
    row_key = lambda row: (int(row["drop"]), int(row["tti"]), int(row["ue_id"]))

    for raw_pair in pairs:
        if len(raw_pair) != 2:
            lines.append(f"[paired-debug] invalid_pair={raw_pair!r}")
            continue
        case_a, case_b = str(raw_pair[0]), str(raw_pair[1])
        drops = sorted({
            drop for drop, case_id in scheduled_pair_lists
            if case_id in (case_a, case_b)
        })
        set_same = True
        order_same = True
        first_schedule_diff = None
        for drop in drops:
            a = scheduled_pair_lists.get((drop, case_a), [])
            b = scheduled_pair_lists.get((drop, case_b), [])
            if set(a) != set(b):
                set_same = False
            if a != b:
                order_same = False
                if first_schedule_diff is None:
                    pos = next(
                        (i for i in range(max(len(a), len(b)))
                         if i >= len(a) or i >= len(b) or a[i] != b[i]),
                        0,
                    )
                    first_schedule_diff = (
                        int(drop), int(pos),
                        None if pos >= len(a) else a[pos],
                        None if pos >= len(b) else b[pos],
                    )

        rows_a = {row_key(row): row for row in link_rows if row.get("scheme") == case_a}
        rows_b = {row_key(row): row for row in link_rows if row.get("scheme") == case_b}
        keys_a, keys_b = set(rows_a), set(rows_b)
        common_keys = sorted(keys_a & keys_b)
        mismatch_counts = {field: 0 for field in (*float_fields, *exact_fields)}
        max_abs_diff = {field: 0.0 for field in float_fields}
        first_row_diff = None
        for key in common_keys:
            a, b = rows_a[key], rows_b[key]
            for field in comparison_fields:
                if field in float_fields:
                    av, bv = float(a[field]), float(b[field])
                    delta = abs(av - bv)
                    max_abs_diff[field] = max(max_abs_diff[field], delta)
                    differs = not np.isclose(av, bv, rtol=1e-12, atol=1e-12, equal_nan=True)
                else:
                    av, bv = a[field], b[field]
                    differs = av != bv
                if differs:
                    mismatch_counts[field] += 1
                    if first_row_diff is None:
                        first_row_diff = (key, field, av, bv, a, b)

        lines.append(
            f"[paired-debug] pair={case_a}|{case_b} drops={len(drops)} "
            f"schedule_set_equal={int(set_same)} schedule_order_equal={int(order_same)} "
            f"rows_a={len(rows_a)} rows_b={len(rows_b)} aligned_rows={len(common_keys)} "
            f"missing_keys_a={len(keys_b - keys_a)} missing_keys_b={len(keys_a - keys_b)}"
        )
        if first_schedule_diff is not None:
            drop, pos, a_link, b_link = first_schedule_diff
            lines.append(
                f"[paired-debug] schedule_first_diff drop={drop} pos={pos} "
                f"A={a_link} B={b_link}"
            )
        lines.append(
            "[paired-debug] mismatch_counts "
            + " ".join(f"{field}={mismatch_counts[field]}" for field in comparison_fields)
        )
        lines.append(
            "[paired-debug] max_abs_diff "
            + " ".join(f"{field}={max_abs_diff[field]:.12g}" for field in float_fields)
        )
        if not common_keys:
            lines.append("[paired-debug] result=NO_ALIGNED_ROWS")
        elif first_row_diff is None and keys_a == keys_b:
            lines.append("[paired-debug] result=EXACT_MATCH")
        elif first_row_diff is not None:
            (drop, tti, ue_id), field, av, bv, a, b = first_row_diff
            lines.append(
                f"[paired-debug] first_row_diff drop={drop} tti={tti} ue={ue_id} "
                f"field={field} A={av} B={bv} "
                f"posA={a.get('link_position')} posB={b.get('link_position')} "
                f"rngA={float(a.get('ack_random_uniform', float('nan'))):.12g} "
                f"rngB={float(b.get('ack_random_uniform', float('nan'))):.12g}"
            )
        else:
            lines.append("[paired-debug] result=ROW_KEYS_DIFFER")
    lines.append("[paired-debug] end")
    return lines


def schedule_similarity_rows(pair_sets: Dict[Tuple[int, str], set],
                             schemes: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_drop: List[Dict[str, Any]] = []
    drops = sorted({drop for drop, _ in pair_sets})
    for a, b in combinations(schemes, 2):
        for drop in drops:
            sa = pair_sets.get((drop, a), set())
            sb = pair_sets.get((drop, b), set())
            inter = len(sa & sb)
            union = len(sa | sb)
            max_size = max(len(sa), len(sb))
            min_size = min(len(sa), len(sb))
            per_drop.append({
                "drop": int(drop),
                "scheme_a": a,
                "scheme_b": b,
                "num_pairs_a": int(len(sa)),
                "num_pairs_b": int(len(sb)),
                "num_same_pairs": int(inter),
                "num_union_pairs": int(union),
                "jaccard_similarity": float(inter / union) if union else 1.0,
                "same_pair_ratio_over_max_size": float(inter / max_size) if max_size else 1.0,
                "overlap_coefficient": float(inter / min_size) if min_size else (1.0 if max_size == 0 else 0.0),
                "exact_schedule_match": int(sa == sb),
            })

    aggregate: List[Dict[str, Any]] = []
    for a, b in combinations(schemes, 2):
        rows = [r for r in per_drop if r["scheme_a"] == a and r["scheme_b"] == b]
        sum_inter = sum(int(r["num_same_pairs"]) for r in rows)
        sum_union = sum(int(r["num_union_pairs"]) for r in rows)
        aggregate.append({
            "scheme_a": a,
            "scheme_b": b,
            "num_drops": int(len(rows)),
            "mean_jaccard_similarity": float(np.mean([r["jaccard_similarity"] for r in rows])) if rows else 0.0,
            "micro_jaccard_similarity": float(sum_inter / sum_union) if sum_union else 1.0,
            "mean_same_pair_ratio_over_max_size": float(np.mean([r["same_pair_ratio_over_max_size"] for r in rows])) if rows else 0.0,
            "exact_schedule_match_ratio": float(np.mean([r["exact_schedule_match"] for r in rows])) if rows else 0.0,
            "total_same_pairs": int(sum_inter),
            "total_union_pairs": int(sum_union),
        })
    return per_drop, aggregate


def summarize_su_snr(samples: List[Dict[str, Any]], schemes: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    maxima: List[Dict[str, Any]] = []
    by_ue: Dict[Tuple[int, str, int], List[float]] = {}
    for r in samples:
        by_ue.setdefault((int(r["drop"]), str(r["scheme"]), int(r["ue_id"])), []).append(float(r["su_snr_db"]))
    for (drop, scheme, ue_id), vals in sorted(by_ue.items()):
        maxima.append({
            "drop": drop,
            "scheme": scheme,
            "ue_id": ue_id,
            "max_su_snr_db": float(max(vals)),
            "num_reported_candidates": int(len(vals)),
        })

    summary: List[Dict[str, Any]] = []
    for scheme in schemes:
        vals = [float(r["su_snr_db"]) for r in samples if r["scheme"] == scheme]
        max_vals = [float(r["max_su_snr_db"]) for r in maxima if r["scheme"] == scheme]
        summary.append({
            "scheme": scheme,
            "num_reported_candidate_samples": int(len(vals)),
            "avg_reported_su_snr_db": float(np.mean(vals)) if vals else 0.0,
            "p05_reported_su_snr_db": percentile(vals, 5.0),
            "p50_reported_su_snr_db": percentile(vals, 50.0),
            "num_ue_samples": int(len(max_vals)),
            "avg_max_su_snr_per_ue_db": float(np.mean(max_vals)) if max_vals else 0.0,
            "p05_max_su_snr_per_ue_db": percentile(max_vals, 5.0),
            "p50_max_su_snr_per_ue_db": percentile(max_vals, 50.0),
        })
    return maxima, summary


def run_simulation(cfg: Dict[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    if str(cfg.get("measurement", {}).get("gamma_backend", "numpy")).lower() in ("cupy", "gpu", "cuda", "auto"):
        # Sionna/TensorFlow and CuPy may share the same GPU in one process. Ask
        # TensorFlow to grow its allocation instead of reserving all VRAM before
        # CuPy starts the Gamma kernels. Respect an explicit user setting.
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    evaluation_plan = resolve_evaluation_plan(cfg)
    evaluation_cases = evaluation_plan.cases
    case_ids = evaluation_plan.case_ids
    feedback_schemes = evaluation_plan.feedback_schemes
    cases_by_id = evaluation_plan.cases_by_id
    cfg.setdefault("_resolved", {})["evaluation_cases"] = [case.to_dict() for case in evaluation_cases]
    cfg["_resolved"]["evaluation_references"] = dict(evaluation_plan.references)
    noise_bw_hz = occupied_bandwidth_hz(cfg)
    cfg["_resolved"]["occupied_bandwidth_hz"] = noise_bw_hz
    cfg["_resolved"]["occupied_bandwidth_mhz"] = noise_bw_hz / 1e6
    save_config(out_dir / "resolved_config.yaml", cfg)
    _progress(
        cfg,
        "[init] occupied bandwidth="
        f"{noise_bw_hz / 1e6:.3f} MHz "
        "(derived from pdsch.num_prbs and system.subcarrier_spacing_khz)",
    )

    if cfg.get("sionna", {}).get("enable_import_probe", True):
        write_json(out_dir / "sionna_import_probe.json", SionnaImportProbe().run())

    link_adapter = make_link_adapter(cfg)
    scheduler_link_adapter = make_scheduler_link_adapter(link_adapter, cfg)
    link_status = dict(link_adapter.status.__dict__)
    if scheduler_link_adapter is not link_adapter and hasattr(scheduler_link_adapter, "status"):
        link_status["scheduler_lookup"] = dict(scheduler_link_adapter.status)
    write_json(out_dir / "link_abstraction_status.json", link_status)
    if scheduler_link_adapter is not link_adapter:
        lookup_status = scheduler_link_adapter.status
        _progress(
            cfg,
            "[init] scheduler link lookup ready, "
            f"regions={lookup_status['num_decision_regions']}, "
            f"elapsed={lookup_status['build_elapsed_s']:.3f}s",
        )

    rng_master = np.random.default_rng(int(cfg["system"].get("random_seed", 1)))
    tx_cfg = ArrayConfig.from_dict(cfg["tx_array"])
    rx_cfg = ArrayConfig.from_dict(cfg["ue_array"])
    rf_arch = resolve_rf_architecture(cfg, tx_cfg)
    effective_mu_order = resolved_max_mu_order(cfg, rf_arch)
    cfg.setdefault("_resolved", {})["max_mu_order"] = effective_mu_order
    cfg["_resolved"]["rf_architecture"] = rf_arch.to_dict()
    _progress(cfg, f"[init] RF={rf_arch.connectivity}, tx_units/TRP={rf_arch.tx_units_per_trp}, max_mu_order={effective_mu_order}")
    write_json(out_dir / "rf_architecture_summary.json", rf_arch.to_dict())
    write_json(out_dir / "array_config_summary.json", {
        "tx_array": tx_cfg.to_dict(),
        "ue_array": rx_cfg.to_dict(),
        "tx_ae_matches_requested": tx_cfg.num_ant == int(cfg["tx_array"].get("num_ae", tx_cfg.num_ant)),
        "tx_txru_matches_requested": (tx_cfg.num_txru is None) or (tx_cfg.num_txru == int(cfg["tx_array"].get("num_txru", tx_cfg.num_txru))),
        "ue_ae_matches_requested": rx_cfg.num_ant == int(cfg["ue_array"].get("num_ae", rx_cfg.num_ant)),
        "ue_rxru_matches_requested": (rx_cfg.num_txru is None) or (rx_cfg.num_txru == int(cfg["ue_array"].get("num_rxru", rx_cfg.num_txru))),
        "tx_beams_per_codebook": _max_beams_from_cfg(cfg["tx_array"], tx_cfg),
        "tx_effective_tx_units_per_sector": _panels_per_cell(cfg, tx_cfg, rf_arch),
        "tx_total_beams_per_sector": _panels_per_cell(cfg, tx_cfg, rf_arch) * max(0, int(_max_beams_from_cfg(cfg["tx_array"], tx_cfg) or (tx_cfg.configured_beams_per_codebook or 0))),
        "ue_rx_beams": _max_beams_from_cfg(cfg["ue_array"], rx_cfg),
        "rf_architecture": rf_arch.to_dict(),
        "resolved_max_mu_order": effective_mu_order,
    })

    # Build a representative topology once to determine network cells and draw topology.
    topo0 = make_topology(cfg, np.random.default_rng(int(cfg["system"].get("random_seed", 1))))
    site_id_by_cell = [topo0.sector_by_cell(c).site_id for c in range(topo0.num_cells)]
    beam_ids, tx_beams = build_network_tx_beams(
        num_cells=topo0.num_cells,
        panels_per_cell=_panels_per_cell(cfg, tx_cfg, rf_arch),
        tx_cfg=tx_cfg,
        max_beams_per_panel=_max_beams_from_cfg(cfg["tx_array"], tx_cfg),
        site_id_by_cell=site_id_by_cell,
        rf_architecture=rf_arch,
    )
    rx_beams = dft_codebook_from_array(rx_cfg,
                                       max_beams=_max_beams_from_cfg(cfg["ue_array"], rx_cfg))

    total_tx_power_w = float(dbm_to_watt(float(cfg["system"]["tx_power_dbm"])))
    num_tx_units = max(1, topo0.num_cells * _panels_per_cell(cfg, tx_cfg, rf_arch))
    power_mode = str(cfg["trp"].get("panel_power_mode", "per_tx_unit_equal")).lower()
    if power_mode in ("per_panel_equal", "per_tx_unit_equal", "per_txru_equal"):
        tx_power_w_per_panel = total_tx_power_w / num_tx_units
    else:
        tx_power_w_per_panel = total_tx_power_w
    noise_w = thermal_noise_watt(noise_bw_hz,
                                 noise_density_dbm_per_hz=float(cfg["noise"].get("thermal_noise_density_dbm_per_hz", -174.0)),
                                 noise_figure_db=float(cfg["noise"].get("ue_noise_figure_db", 7.0)))

    all_link_rows: List[LinkEvalRow] = []
    schedule_rows: List[Dict[str, Any]] = []
    drop_rows: List[Dict[str, Any]] = []
    ue_rows: List[Dict[str, Any]] = []
    site_rows_all: List[Dict[str, Any]] = []
    sector_rows_all: List[Dict[str, Any]] = []
    beam_rows: List[Dict[str, Any]] = [b.to_dict() for b in beam_ids]
    report_rows: List[Dict[str, Any]] = []
    su_snr_sample_rows: List[Dict[str, Any]] = []
    scheduled_ue_su_throughput_rows: List[Dict[str, Any]] = []
    scheduler_stat_rows: List[Dict[str, Any]] = []
    scheduler_iteration_rows: List[Dict[str, Any]] = []
    runtime_phase_rows: List[Dict[str, Any]] = []
    if scheduler_link_adapter is not link_adapter:
        runtime_phase_rows.append({
            "drop": -1,
            "case_id": "all",
            "phase": "scheduler_link_lookup_build",
            "elapsed_s": float(scheduler_link_adapter.status["build_elapsed_s"]),
            **_counter_snapshot(scheduler_link_adapter),
        })
    tbar_by_scheme: Dict[str, Dict[int, float]] = {}
    channel_backend_rows: List[Dict[str, Any]] = []
    gamma_backend_rows: List[Dict[str, Any]] = []
    scheduled_pair_sets: Dict[Tuple[int, str], set] = {}
    scheduled_pair_lists: Dict[Tuple[int, str], List[Tuple[int, int]]] = {}
    domain_mode = normalize_domain_mode(cfg["scheduler"].get("domain_mode", "global"))

    baseline_reference = evaluation_plan.references.get("baseline")
    upper_bound_enabled = bool(
        cfg.get("analysis", {}).get("baseline_no_interference_upper_bound", True)
        and baseline_reference is not None
    )
    analysis_schemes = list(case_ids)
    if upper_bound_enabled:
        analysis_schemes.append(BASELINE_NO_INTERFERENCE_SCHEME)
    total_ues = int(cfg["ue_drop"]["num_ut_per_sector"]) * max(1, topo0.num_cells)
    for s in case_ids:
        tbar_by_scheme[s] = {u: float(cfg["scheduler"].get("pf_tbar_init_mbps", 1.0))
                             for u in range(total_ues)}

    # Topology plot for the first representative drop.
    try:
        plot_topology(topo0, cfg, out_dir / "figures" / "topology.png")
    except Exception as e:
        write_json(out_dir / "topology_plot_error.json", {"error": f"{type(e).__name__}: {e}"})

    num_drops = int(cfg["system"].get("num_drops", 1))
    num_tti = int(cfg["system"].get("num_tti_per_drop", 1))
    olla_warmup_tti = int(cfg["link_abstraction"].get("olla_warmup_tti", 0))
    _progress(cfg, f"[run] drops={num_drops}, warmup_tti/drop={olla_warmup_tti}, measured_tti/drop={num_tti}, cases={','.join(case_ids)}, beams={len(beam_ids)}, tx_units/sector={_panels_per_cell(cfg, tx_cfg, rf_arch)}")
    for drop in range(num_drops):
        _progress(cfg, f"[drop {drop+1}/{num_drops}] topology + channel generation")
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        topo = make_topology(cfg, rng)
        ch = generate_channel(topo, cfg, tx_cfg, rx_cfg, rng)
        channel_backend_rows.append({"drop": drop, "backend": ch.backend, "backend_status": ch.backend_status})
        _progress(cfg, f"[drop {drop+1}/{num_drops}] channel backend={ch.backend}; computing Gamma measurement")
        ue_site_ids = {int(ue.ue_id): int(ue.site_id) for ue in topo.ues}
        ue_serving_cells = {int(ue.ue_id): int(ue.serving_cell) for ue in topo.ues}
        candidate_beam_indices_by_ue = _domain_candidate_beam_indices_by_ue(topo, beam_ids, domain_mode)
        meas = compute_gamma_measurement(ch.h_freq, tx_beams, rx_beams, beam_ids,
                                         tx_power_w_per_panel=tx_power_w_per_panel,
                                         noise_power_w=noise_w,
                                         candidate_beam_indices_by_ue=candidate_beam_indices_by_ue,
                                         compute_backend=cfg["measurement"].get("gamma_backend", "numpy"),
                                         ue_batch_size=cfg["measurement"].get("gamma_ue_batch_size", 0))
        gamma_backend_rows.append({
            "drop": int(drop),
            "requested_backend": str(cfg["measurement"].get("gamma_backend", "numpy")),
            "effective_backend": meas.compute_backend,
            "backend_status": meas.backend_status,
            "ue_batch_size": int(cfg["measurement"].get("gamma_ue_batch_size", 0)),
            "elapsed_s": float(meas.elapsed_s),
        })
        _progress(cfg, f"[drop {drop+1}/{num_drops}] Gamma backend={meas.compute_backend}, elapsed={meas.elapsed_s:.3f}s; {meas.backend_status}")
        feedback_started = perf_counter()
        feedback_counters_before = _counter_snapshot(scheduler_link_adapter)
        reports_by_scheme = make_reports(
            meas, beam_ids, schemes=feedback_schemes,
            k1=int(cfg["feedback"].get("service_beam_top_k1", 4)),
            oracle_top_k=int(cfg["feedback"].get("oracle_service_beam_top_k", 4)),
            k2=int(cfg["feedback"].get("conflict_top_k2", 3)),
            threshold_db=float(cfg["feedback"].get("conflict_sinr_threshold_db", 0.0)),
            ue_site_ids=ue_site_ids,
            ue_serving_cells=ue_serving_cells,
            candidate_beam_indices_by_ue=candidate_beam_indices_by_ue,
            link_adapter=scheduler_link_adapter,
        )
        feedback_elapsed_s = float(perf_counter() - feedback_started)
        feedback_counter_delta = _counter_delta(
            feedback_counters_before, _counter_snapshot(scheduler_link_adapter), prefix="",
        )
        runtime_phase_rows.append({
            "drop": int(drop),
            "case_id": "all",
            "phase": "feedback_generation",
            "elapsed_s": feedback_elapsed_s,
            **feedback_counter_delta,
        })
        site_rows, sector_rows = topology_to_rows(topo)
        for r in site_rows:
            rr = dict(r); rr["drop"] = drop; site_rows_all.append(rr)
        for r in sector_rows:
            rr = dict(r); rr["drop"] = drop; sector_rows_all.append(rr)
        for ue in topo.ues:
            d = ue.to_dict()
            d["drop"] = drop
            d["pathloss_db"] = float(ch.pathloss_db[ue.ue_id]) if ue.ue_id < len(ch.pathloss_db) else float("nan")
            d["shadow_db"] = float(ch.shadow_db[ue.ue_id]) if ue.ue_id < len(ch.shadow_db) else float("nan")
            ue_rows.append(d)
        for scheme, reps in reports_by_scheme.items():
            for r in reps:
                report_rows.append({"drop": drop, "scheme": scheme, "ue_id": r.ue_id,
                                    "report_json": json.dumps(r.to_dict(beam_ids), ensure_ascii=False)})
                for rank, cand in enumerate(r.candidates):
                    su_snr_sample_rows.append({
                        "drop": int(drop),
                        "scheme": scheme,
                        "ue_id": int(r.ue_id),
                        "candidate_rank": int(rank),
                        "beam_index": int(cand.beam_index),
                        "beam_id": beam_ids[cand.beam_index].short(),
                        "su_snr_db": float(cand.su_snr_db),
                        "su_mcs": int(cand.su_mcs),
                        "su_outage": bool(cand.su_outage),
                    })

        _progress(cfg, f"[drop {drop+1}/{num_drops}] scheduling {len(evaluation_cases)} evaluation cases")
        # Drops are independent UE/topology realizations. Keep OLLA state across
        # TTIs within a drop, but do not leak offsets to the next random drop.
        olla_state: Dict[Tuple[str, int], float] = {}
        # Every case starts from the same ACK random stream for cleaner paired
        # comparisons. OLLA state remains isolated by case_id.
        common_link_rng_state = copy.deepcopy(rng.bit_generator.state)
        for case_index, case in enumerate(evaluation_cases, start=1):
            scheme = case.feedback_scheme
            case_id = case.case_id
            case_prefix = f"[drop {drop+1}/{num_drops}][case {case_index}/{len(evaluation_cases)}] {case_id}"
            _progress(cfg, f"{case_prefix} started")
            scheduler_started = perf_counter()
            scheduler_counters_before = _counter_snapshot(scheduler_link_adapter)
            sched = schedule(
                reports_by_scheme[scheme], beam_ids, cfg, tbar_by_scheme.get(case_id),
                link_adapter=scheduler_link_adapter, algorithm=case.algorithm,
                progress_callback=lambda message, prefix=case_prefix: _progress(cfg, f"{prefix} {message}"),
            )
            scheduler_elapsed_s = float(perf_counter() - scheduler_started)
            scheduler_counter_delta = _counter_delta(
                scheduler_counters_before, _counter_snapshot(scheduler_link_adapter),
                prefix="scheduler_",
            )
            sched.case_id = case_id
            sched.feedback_scheme = scheme
            sched.algorithm = case.algorithm
            sched.metadata["algorithm"] = case.algorithm
            case_metrics = {
                "scheduler_elapsed_s": scheduler_elapsed_s,
                **scheduler_counter_delta,
            }
            _attach_case_scheduler_metrics(sched, case_metrics)
            _progress(cfg, f"{case_prefix} scheduler finished, elapsed={scheduler_elapsed_s:.3f}s")
            scheduler_iteration_rows.extend(_schedule_iteration_rows(drop, case_id, sched))
            runtime_phase_rows.append({
                "drop": int(drop),
                "case_id": case_id,
                "phase": "scheduler",
                "elapsed_s": scheduler_elapsed_s,
                **scheduler_counter_delta,
            })
            scheduled_pair_sets[(drop, case_id)] = {
                (int(link.ue_id), int(link.beam_index)) for link in sched.links
            }
            scheduled_pair_lists[(drop, case_id)] = [
                (int(link.ue_id), int(link.beam_index)) for link in sched.links
            ]
            schedule_rows.append({"drop": drop, "scheme": case_id,
                                  "case_id": case_id,
                                  "feedback_scheme": scheme,
                                  "algorithm": case.algorithm,
                                  "objective_value": sched.objective_value,
                                  "num_scheduled": len(sched.links),
                                  "num_outage": int(sum(link.predicted_outage for link in sched.links)),
                                  "outage_occurred": bool(any(link.predicted_outage for link in sched.links)),
                                  "schedule_json": json.dumps(sched.to_dict(beam_ids), ensure_ascii=False)})
            scheduled_ue_su_throughput_rows.extend(
                build_scheduled_ue_su_throughput_rows(
                    drop, case_id, sched, reports_by_scheme[scheme], beam_ids,
                    scheduler_link_adapter,
                )
            )
            case_rng = np.random.default_rng()
            case_rng.bit_generator.state = copy.deepcopy(common_link_rng_state)
            rng_state_before_link_eval = copy.deepcopy(case_rng.bit_generator.state)
            link_eval_started = perf_counter()
            link_counters_before = _counter_snapshot(link_adapter)
            link_rows, olla_state = run_tti_loop(sched, ch.h_freq, tx_beams, rx_beams, beam_ids, meas,
                                                 tx_power_w_per_panel, cfg, drop, case_rng, olla_state,
                                                 link_adapter=link_adapter)
            link_eval_elapsed_s = float(perf_counter() - link_eval_started)
            link_counter_delta = _counter_delta(
                link_counters_before, _counter_snapshot(link_adapter), prefix="link_evaluation_",
            )
            _attach_case_scheduler_metrics(sched, {
                "link_evaluation_elapsed_s": link_eval_elapsed_s,
                **link_counter_delta,
            })
            scheduler_stat_rows.extend(_schedule_stat_rows(drop, case_id, sched))
            runtime_phase_rows.append({
                "drop": int(drop),
                "case_id": case_id,
                "phase": "link_evaluation",
                "elapsed_s": link_eval_elapsed_s,
                **link_counter_delta,
            })
            _progress(cfg, f"{case_prefix} finished, link_eval_elapsed={link_eval_elapsed_s:.3f}s")
            all_link_rows.extend(link_rows)
            if case_id == baseline_reference and upper_bound_enabled:
                upper_sched = ScheduleResult(
                    scheme=BASELINE_NO_INTERFERENCE_SCHEME,
                    objective_value=float(sched.objective_value),
                    links=list(sched.links),
                    metadata={
                        "source_scheme": case_id,
                        "interference_mode": "forced_zero",
                    },
                    case_id=BASELINE_NO_INTERFERENCE_SCHEME,
                    feedback_scheme=BASELINE_NO_INTERFERENCE_SCHEME,
                    algorithm=case.algorithm,
                )
                upper_rng = np.random.default_rng()
                upper_rng.bit_generator.state = rng_state_before_link_eval
                upper_rows, _ = run_tti_loop(
                    upper_sched, ch.h_freq, tx_beams, rx_beams, beam_ids, meas,
                    tx_power_w_per_panel, cfg, drop, upper_rng, {},
                    link_adapter=link_adapter, ignore_interference=True,
                )
                all_link_rows.extend(upper_rows)
            if cfg["scheduler"].get("objective", "sum_rate") == "proportional_fair":
                alpha = 0.05
                for r in link_rows:
                    old = tbar_by_scheme[case_id].get(r.ue_id, float(cfg["scheduler"].get("pf_tbar_init_mbps", 1.0)))
                    tbar_by_scheme[case_id][r.ue_id] = (1 - alpha) * old + alpha * r.goodput_mbps

        _progress(cfg, f"[drop {drop+1}/{num_drops}] finished")
        drop_rows.append({
            "drop": drop,
            "num_ues": len(topo.ues),
            "num_cells": topo.num_cells,
            "num_sites": len(topo.sites),
            "num_beams": len(beam_ids),
            "olla_warmup_tti": olla_warmup_tti,
            "num_measured_tti": num_tti,
            "scheduler_domain_mode": domain_mode,
            "occupied_bandwidth_mhz": float(noise_bw_hz / 1e6),
            "noise_dbm": float(watt_to_dbm(noise_w)),
            "tx_power_per_tx_unit_dbm": float(watt_to_dbm(tx_power_w_per_panel)),
            "avg_su_snr_db": _domain_metric(meas.su_snr_db, candidate_beam_indices_by_ue, np.mean),
            "p95_su_snr_db": _domain_metric(meas.su_snr_db, candidate_beam_indices_by_ue, lambda x: np.percentile(x, 95)),
            "avg_measurement_beams_per_ue": _avg_domain_beams(candidate_beam_indices_by_ue, len(beam_ids), len(topo.ues)),
            "channel_backend": ch.backend,
            "link_adaptation_backend": link_adapter.status.backend,
        })

    link_dict_rows = _asdict_rows(all_link_rows)
    paired_debug_cfg = cfg.get("analysis", {}).get("paired_case_debug", {}) or {}
    if bool(paired_debug_cfg.get("enabled", False)):
        configured_pairs = paired_debug_cfg.get("pairs", []) or []
        debug_lines = build_paired_case_debug_lines(
            scheduled_pair_lists, link_dict_rows,
            [[str(value) for value in pair] for pair in configured_pairs],
        )
        debug_text = "\n".join(debug_lines) + "\n"
        debug_path = ensure_dir(out_dir / "metrics") / "paired_case_debug.txt"
        debug_path.write_text(
            debug_text, encoding="utf-8",
        )
        print(debug_text, end="", flush=True)
    write_csv(out_dir / "metrics" / "link_tti.csv", link_dict_rows)
    write_csv(out_dir / "metrics" / "schedules.csv", schedule_rows)
    write_csv(out_dir / "metrics" / "drops.csv", drop_rows)
    write_csv(out_dir / "metrics" / "ues.csv", ue_rows)
    write_csv(out_dir / "metrics" / "sites.csv", site_rows_all)
    write_csv(out_dir / "metrics" / "sectors.csv", sector_rows_all)
    write_csv(out_dir / "metrics" / "beams.csv", beam_rows)
    write_csv(out_dir / "metrics" / "gamma_measurement_backend.csv", gamma_backend_rows)
    write_csv(out_dir / "metrics" / "reports.csv", report_rows)
    write_csv(out_dir / "metrics" / "scheduler_stats.csv", scheduler_stat_rows)
    write_csv(out_dir / "metrics" / "scheduler_iterations.csv", scheduler_iteration_rows)
    write_csv(out_dir / "metrics" / "runtime_phases.csv", runtime_phase_rows)
    write_csv(out_dir / "metrics" / "channel_backend.csv", channel_backend_rows)

    ue_goodput_rows = build_ue_goodput_rows(
        link_dict_rows, analysis_schemes, ue_rows, num_tti,
    )
    write_csv(out_dir / "metrics" / "ue_goodput.csv", ue_goodput_rows)

    similarity_by_drop, similarity_summary = schedule_similarity_rows(scheduled_pair_sets, case_ids)
    write_csv(out_dir / "metrics" / "schedule_similarity_by_drop.csv", similarity_by_drop)
    write_csv(out_dir / "metrics" / "schedule_similarity.csv", similarity_summary)

    su_snr_max_rows, su_snr_summary = summarize_su_snr(su_snr_sample_rows, feedback_schemes)
    write_csv(out_dir / "metrics" / "su_snr_samples.csv", su_snr_sample_rows)
    write_csv(out_dir / "metrics" / "su_snr_max_per_ue.csv", su_snr_max_rows)
    write_csv(out_dir / "metrics" / "su_snr_summary.csv", su_snr_summary)
    scheduled_ue_su_throughput_summary = summarize_scheduled_ue_su_throughput(
        scheduled_ue_su_throughput_rows, case_ids,
    )
    write_csv(
        out_dir / "metrics" / "scheduled_ue_su_throughput.csv",
        scheduled_ue_su_throughput_rows,
    )
    write_csv(
        out_dir / "metrics" / "scheduled_ue_su_throughput_summary.csv",
        scheduled_ue_su_throughput_summary,
    )

    summary = summarize_results(
        link_dict_rows, analysis_schemes, ue_goodput_rows,
        references=evaluation_plan.references, cases_by_id=cases_by_id,
    )
    summary["_backend"] = {
        "link_adaptation": link_adapter.status.__dict__,
        "channel_backends": channel_backend_rows,
    }
    summary["_schedule_similarity"] = similarity_summary
    summary["_su_snr"] = su_snr_summary
    summary["_scheduled_ue_su_throughput"] = scheduled_ue_su_throughput_summary
    write_json(out_dir / "metrics" / "summary.json", summary)
    write_csv(out_dir / "metrics" / "summary.csv", [{"scheme": k, **v} for k, v in summary.items() if isinstance(v, dict) and not k.startswith("_")])

    make_plots(
        out_dir, link_dict_rows, summary, analysis_schemes,
        ue_goodput_rows=ue_goodput_rows,
        su_snr_samples=su_snr_sample_rows,
        su_snr_max_rows=su_snr_max_rows,
        scheduled_ue_su_throughput_rows=scheduled_ue_su_throughput_rows,
        feedback_schemes=feedback_schemes,
    )

    if cfg.get("coverage_heatmap", {}).get("enabled", True):
        _progress(cfg, "[coverage] generating coverage heatmap and fixed-vertical-beam CDF")
        try:
            x, y, p_dbm, best_beam, heat_status = compute_coverage_heatmap_standard_sampling(
                cfg, tx_cfg, rx_cfg, tx_beams, rx_beams, beam_ids,
                tx_power_w_per_panel, noise_w, np.random.default_rng(int(cfg["system"].get("random_seed", 1)) + 999))
            plot_heatmap(x, y, p_dbm, out_dir / "figures" / "coverage_heatmap.png",
                         title="Coverage heatmap by configured channel sampling")
            plot_best_beam_heatmap(x, y, best_beam, out_dir / "figures" / "best_beam_heatmap.png")
            heat_rows = []
            for iy, yy in enumerate(y):
                for ix, xx in enumerate(x):
                    if not np.isnan(p_dbm[iy, ix]):
                        heat_rows.append({"x_m": float(xx), "y_m": float(yy),
                                          "best_beam_index": int(best_beam[iy, ix]),
                                          "best_rx_power_dbm": float(p_dbm[iy, ix])})
            write_csv(out_dir / "metrics" / "coverage_heatmap.csv", heat_rows)
            write_json(out_dir / "coverage_heatmap_status.json", {"backend_status": heat_status})

            if cfg.get("coverage_heatmap", {}).get("fixed_vertical_beam_cdf", {}).get("enabled", False):
                v_summary, v_samples, selected_v = compute_fixed_vertical_beam_cdf(
                    cfg, tx_cfg, rx_cfg, rx_beams, site_id_by_cell,
                    tx_power_w_per_panel, noise_w,
                    np.random.default_rng(int(cfg["system"].get("random_seed", 1)) + 1999),
                    panels_per_cell=_panels_per_cell(cfg, tx_cfg, rf_arch),
                    rf_architecture=rf_arch,
                )
                write_csv(out_dir / "metrics" / "fixed_vertical_beam_summary.csv", v_summary)
                write_csv(out_dir / "metrics" / "fixed_vertical_beam_samples.csv", v_samples)
                plot_cdf({r["label"]: r["values_dbm"] for r in v_summary},
                         "Coverage RSRP averaged over horizontal scan beams [dBm]",
                         f"Fixed vertical beam coverage CDF (selected v={selected_v})",
                         out_dir / "figures" / "fixed_vertical_beam_cdf.png")
                write_json(out_dir / "fixed_vertical_beam_selection.json", {"selected_v_index": selected_v})
        except Exception as e:
            write_json(out_dir / "coverage_heatmap_error.json", {"error": f"{type(e).__name__}: {e}"})

    _progress(cfg, f"[done] outputs written to {out_dir.resolve()}")
    return summary


def summarize_results(link_rows: List[Dict[str, Any]],
                      schemes: List[str],
                      ue_goodput_rows: List[Dict[str, Any]] | None = None,
                      references: Dict[str, str] | None = None,
                      cases_by_id: Dict[str, EvaluationCase] | None = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for scheme in schemes:
        rows = [r for r in link_rows if r.get("scheme") == scheme]
        if not rows:
            summary[scheme] = {
                "avg_system_goodput_mbps": 0.0,
                "avg_ue_goodput_mbps": 0.0,
                "p05_ue_goodput_mbps": 0.0,
                "avg_eff_sinr_db": 0.0,
                "avg_tbler": 0.0,
                "num_tx": 0,
            }
            continue
        sys_by_slot: Dict[Tuple[int, int], float] = {}
        for r in rows:
            key = (int(r["drop"]), int(r["tti"]))
            sys_by_slot[key] = sys_by_slot.get(key, 0.0) + float(r["goodput_mbps"])
        if ue_goodput_rows is None:
            ue_vals: Dict[Tuple[int, int], List[float]] = {}
            for r in rows:
                ue_key = (int(r["drop"]), int(r["ue_id"]))
                ue_vals.setdefault(ue_key, []).append(float(r["goodput_mbps"]))
            ue_mean = [float(np.mean(v)) for v in ue_vals.values()]
        else:
            ue_mean = [
                float(r["avg_goodput_mbps"])
                for r in ue_goodput_rows
                if r.get("scheme") == scheme
            ]
        case = (cases_by_id or {}).get(scheme)
        summary[scheme] = {
            "case_id": scheme,
            "feedback_scheme": case.feedback_scheme if case is not None else scheme,
            "algorithm": case.algorithm if case is not None else None,
            "avg_system_goodput_mbps": float(np.mean(list(sys_by_slot.values()))) if sys_by_slot else 0.0,
            "avg_ue_goodput_mbps": float(np.mean(ue_mean)) if ue_mean else 0.0,
            "p05_ue_goodput_mbps": percentile(ue_mean, 5.0),
            "avg_eff_sinr_db": float(np.mean([float(r["effective_sinr_db"]) for r in rows])),
            "avg_tbler": float(np.mean([float(r["tbler"]) for r in rows])),
            "ack_rate": float(np.mean([float(r["ack"]) for r in rows])),
            "num_tx": len(rows),
        }
    refs = references or {}
    oracle_case = refs.get("oracle", "full_gamma" if "full_gamma" in summary else None)
    baseline_case = refs.get("baseline", "baseline" if "baseline" in summary else None)
    if oracle_case in summary:
        oracle = max(float(summary[oracle_case].get("avg_system_goodput_mbps", 0.0)), 1e-12)
        for scheme in schemes:
            if isinstance(summary.get(scheme), dict):
                summary[scheme]["oracle_ratio"] = float(summary[scheme].get("avg_system_goodput_mbps", 0.0)) / oracle
    if baseline_case in summary:
        base = float(summary[baseline_case].get("avg_system_goodput_mbps", 0.0))
        base_p05 = float(summary[baseline_case].get("p05_ue_goodput_mbps", 0.0))
        for scheme in schemes:
            if isinstance(summary.get(scheme), dict):
                value = float(summary[scheme].get("avg_system_goodput_mbps", 0.0))
                summary[scheme]["gain_over_baseline"] = (
                    (value - base) / base if base > 0.0 else (0.0 if scheme == baseline_case else None)
                )
                p05 = float(summary[scheme].get("p05_ue_goodput_mbps", 0.0))
                summary[scheme]["p05_gain_over_baseline_mbps"] = p05 - base_p05
                summary[scheme]["p05_gain_over_baseline"] = (
                    (p05 - base_p05) / base_p05 if base_p05 > 0.0 else (0.0 if scheme == baseline_case else None)
                )
    if BASELINE_NO_INTERFERENCE_SCHEME in summary and "baseline" in summary:
        upper = max(float(summary[BASELINE_NO_INTERFERENCE_SCHEME].get("avg_system_goodput_mbps", 0.0)), 1e-12)
        for scheme in schemes:
            if isinstance(summary.get(scheme), dict):
                summary[scheme]["ratio_to_baseline_no_interference_upper_bound"] = (
                    float(summary[scheme].get("avg_system_goodput_mbps", 0.0)) / upper
                )
    return summary


def make_plots(out_dir: Path,
               link_rows: List[Dict[str, Any]],
               summary: Dict[str, Any],
               schemes: List[str],
               ue_goodput_rows: List[Dict[str, Any]] | None = None,
               su_snr_samples: List[Dict[str, Any]] | None = None,
               su_snr_max_rows: List[Dict[str, Any]] | None = None,
               scheduled_ue_su_throughput_rows: List[Dict[str, Any]] | None = None,
               feedback_schemes: List[str] | None = None) -> None:
    sinr_by_scheme = {s: [float(r["effective_sinr_db"]) for r in link_rows if r.get("scheme") == s] for s in schemes}
    goodput_by_scheme = {s: [float(r["goodput_mbps"]) for r in link_rows if r.get("scheme") == s] for s in schemes}
    plot_cdf(sinr_by_scheme, "Effective SINR [dB]", "Post-SINR CDF", out_dir / "figures" / "effective_sinr_cdf.png")
    plot_cdf(goodput_by_scheme, "TTI link goodput [Mbps]", "Link goodput CDF", out_dir / "figures" / "link_goodput_cdf.png")
    if ue_goodput_rows is not None:
        ue_goodput_by_scheme = {
            s: [float(r["avg_goodput_mbps"]) for r in ue_goodput_rows if r.get("scheme") == s]
            for s in schemes
        }
        plot_cdf(
            ue_goodput_by_scheme,
            "Per-UE average goodput [Mbps]",
            "Per-UE goodput CDF (unscheduled UEs included as zero)",
            out_dir / "figures" / "ue_goodput_cdf.png",
        )
    report_schemes = feedback_schemes or schemes
    if su_snr_samples is not None:
        plot_cdf(
            {
                s: [float(r["su_snr_db"]) for r in su_snr_samples if r.get("scheme") == s]
                for s in report_schemes
            },
            "Reported standalone SNR [dB]",
            "Reported SU SNR CDF (every candidate counted)",
            out_dir / "figures" / "reported_su_snr_cdf.png",
        )
    if su_snr_max_rows is not None:
        plot_cdf(
            {
                s: [float(r["max_su_snr_db"]) for r in su_snr_max_rows if r.get("scheme") == s]
                for s in report_schemes
            },
            "Maximum reported standalone SNR per UE [dB]",
            "Per-UE maximum SU SNR CDF",
            out_dir / "figures" / "reported_max_su_snr_per_ue_cdf.png",
        )
    if scheduled_ue_su_throughput_rows is not None:
        plot_cdf(
            {
                s: [
                    float(r["su_throughput_mbps"])
                    for r in scheduled_ue_su_throughput_rows
                    if r.get("scheme") == s
                ]
                for s in schemes
            },
            "Scheduled-UE SU throughput [Mbps]",
            "Scheduled-UE standalone throughput CDF",
            out_dir / "figures" / "scheduled_ue_su_throughput_cdf.png",
        )
    bar = {s: float(summary[s]["avg_system_goodput_mbps"]) for s in schemes if s in summary}
    plot_bar(bar, "Average system goodput [Mbps]", "Average system goodput", out_dir / "figures" / "avg_system_goodput.png")
