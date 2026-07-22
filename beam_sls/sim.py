from __future__ import annotations

import copy
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .channel import SionnaImportProbe, generate_channel
from .codebook import ArrayConfig, build_network_tx_beams, dft_codebook_from_array
from .config import save_config
from .coverage import compute_coverage_heatmap_standard_sampling, compute_fixed_vertical_beam_cdf
from .evaluation import EvaluationCase, resolve_evaluation_plan
from .feedback import make_reports
from .link import LinkEvalRow, run_tti_loop
from .link_adaptation import make_link_adapter
from .measurement import compute_gamma_measurement
from .rf import resolve_rf_architecture, resolved_max_mu_order, tx_units_per_sector
from .plotting import plot_bar, plot_best_beam_heatmap, plot_cdf, plot_heatmap, plot_topology
from .scheduler import ScheduleResult, is_sector_domain_mode, is_site_domain_mode, normalize_domain_mode, schedule
from .topology import make_topology, topology_to_rows
from .utils import dbm_to_watt, ensure_dir, percentile, thermal_noise_watt, watt_to_dbm, write_csv, write_json


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
    save_config(out_dir / "resolved_config.yaml", cfg)

    if cfg.get("sionna", {}).get("enable_import_probe", True):
        write_json(out_dir / "sionna_import_probe.json", SionnaImportProbe().run())

    link_adapter = make_link_adapter(cfg)
    write_json(out_dir / "link_abstraction_status.json", link_adapter.status.__dict__)

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
    noise_bw_hz = int(cfg["pdsch"]["num_prbs"]) * 12.0 * float(cfg["system"].get("subcarrier_spacing_khz", 120.0)) * 1e3
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
    scheduler_stat_rows: List[Dict[str, Any]] = []
    tbar_by_scheme: Dict[str, Dict[int, float]] = {}
    channel_backend_rows: List[Dict[str, Any]] = []
    gamma_backend_rows: List[Dict[str, Any]] = []
    scheduled_pair_sets: Dict[Tuple[int, str], set] = {}
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
        reports_by_scheme = make_reports(
            meas, beam_ids, schemes=feedback_schemes,
            k1=int(cfg["feedback"].get("service_beam_top_k1", 4)),
            oracle_top_k=int(cfg["feedback"].get("oracle_service_beam_top_k", 4)),
            k2=int(cfg["feedback"].get("conflict_top_k2", 3)),
            threshold_db=float(cfg["feedback"].get("conflict_sinr_threshold_db", 0.0)),
            ue_site_ids=ue_site_ids,
            ue_serving_cells=ue_serving_cells,
            candidate_beam_indices_by_ue=candidate_beam_indices_by_ue,
            link_adapter=link_adapter,
        )
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
        for case in evaluation_cases:
            scheme = case.feedback_scheme
            case_id = case.case_id
            sched = schedule(
                reports_by_scheme[scheme], beam_ids, cfg, tbar_by_scheme.get(case_id),
                link_adapter=link_adapter, algorithm=case.algorithm,
            )
            sched.case_id = case_id
            sched.feedback_scheme = scheme
            sched.algorithm = case.algorithm
            sched.metadata["algorithm"] = case.algorithm
            scheduled_pair_sets[(drop, case_id)] = {
                (int(link.ue_id), int(link.beam_index)) for link in sched.links
            }
            schedule_rows.append({"drop": drop, "scheme": case_id,
                                  "case_id": case_id,
                                  "feedback_scheme": scheme,
                                  "algorithm": case.algorithm,
                                  "objective_value": sched.objective_value,
                                  "num_scheduled": len(sched.links),
                                  "num_outage": int(sum(link.predicted_outage for link in sched.links)),
                                  "outage_occurred": bool(any(link.predicted_outage for link in sched.links)),
                                  "schedule_json": json.dumps(sched.to_dict(beam_ids), ensure_ascii=False)})
            scheduler_stat_rows.extend(_schedule_stat_rows(drop, case_id, sched))
            case_rng = np.random.default_rng()
            case_rng.bit_generator.state = copy.deepcopy(common_link_rng_state)
            rng_state_before_link_eval = copy.deepcopy(case_rng.bit_generator.state)
            link_rows, olla_state = run_tti_loop(sched, ch.h_freq, tx_beams, rx_beams, beam_ids, meas,
                                                 tx_power_w_per_panel, cfg, drop, case_rng, olla_state,
                                                 link_adapter=link_adapter)
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
            "noise_dbm": float(watt_to_dbm(noise_w)),
            "tx_power_per_tx_unit_dbm": float(watt_to_dbm(tx_power_w_per_panel)),
            "avg_su_snr_db": _domain_metric(meas.su_snr_db, candidate_beam_indices_by_ue, np.mean),
            "p95_su_snr_db": _domain_metric(meas.su_snr_db, candidate_beam_indices_by_ue, lambda x: np.percentile(x, 95)),
            "avg_measurement_beams_per_ue": _avg_domain_beams(candidate_beam_indices_by_ue, len(beam_ids), len(topo.ues)),
            "channel_backend": ch.backend,
            "link_adaptation_backend": link_adapter.status.backend,
        })

    link_dict_rows = _asdict_rows(all_link_rows)
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
    write_json(out_dir / "metrics" / "summary.json", summary)
    write_csv(out_dir / "metrics" / "summary.csv", [{"scheme": k, **v} for k, v in summary.items() if isinstance(v, dict) and not k.startswith("_")])

    make_plots(
        out_dir, link_dict_rows, summary, analysis_schemes,
        ue_goodput_rows=ue_goodput_rows,
        su_snr_samples=su_snr_sample_rows,
        su_snr_max_rows=su_snr_max_rows,
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
    bar = {s: float(summary[s]["avg_system_goodput_mbps"]) for s in schemes if s in summary}
    plot_bar(bar, "Average system goodput [Mbps]", "Average system goodput", out_dir / "figures" / "avg_system_goodput.png")
