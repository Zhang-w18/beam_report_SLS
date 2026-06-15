from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .channel import SionnaImportProbe, generate_channel
from .codebook import ArrayConfig, build_network_tx_beams, dft_codebook_from_array
from .config import save_config
from .coverage import compute_coverage_heatmap_standard_sampling, compute_fixed_vertical_beam_cdf
from .feedback import make_reports
from .link import LinkEvalRow, run_tti_loop
from .link_adaptation import make_link_adapter
from .measurement import compute_gamma_measurement
from .rf import resolve_rf_architecture, resolved_max_mu_order, tx_units_per_sector
from .plotting import plot_bar, plot_best_beam_heatmap, plot_cdf, plot_heatmap, plot_topology
from .scheduler import is_site_domain_mode, schedule
from .topology import make_topology, topology_to_rows
from .utils import dbm_to_watt, ensure_dir, percentile, thermal_noise_watt, watt_to_dbm, write_csv, write_json


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


def run_simulation(cfg: Dict[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir)
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
    scheduler_stat_rows: List[Dict[str, Any]] = []
    tbar_by_scheme: Dict[str, Dict[int, float]] = {}
    channel_backend_rows: List[Dict[str, Any]] = []
    domain_mode = cfg["scheduler"].get("domain_mode", "global")

    schemes = list(cfg["feedback"].get("schemes", ["baseline"]))
    total_ues = int(cfg["ue_drop"]["num_ut_per_sector"]) * max(1, topo0.num_cells)
    for s in schemes:
        tbar_by_scheme[s] = {u: float(cfg["scheduler"].get("pf_tbar_init_mbps", 1.0))
                             for u in range(total_ues)}

    # Topology plot for the first representative drop.
    try:
        plot_topology(topo0, cfg, out_dir / "figures" / "topology.png")
    except Exception as e:
        write_json(out_dir / "topology_plot_error.json", {"error": f"{type(e).__name__}: {e}"})

    num_drops = int(cfg["system"].get("num_drops", 1))
    num_tti = int(cfg["system"].get("num_tti_per_drop", 1))
    _progress(cfg, f"[run] drops={num_drops}, tti/drop={num_tti}, schemes={','.join(schemes)}, beams={len(beam_ids)}, tx_units/sector={_panels_per_cell(cfg, tx_cfg, rf_arch)}")
    for drop in range(num_drops):
        _progress(cfg, f"[drop {drop+1}/{num_drops}] topology + channel generation")
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        topo = make_topology(cfg, rng)
        ch = generate_channel(topo, cfg, tx_cfg, rx_cfg, rng)
        channel_backend_rows.append({"drop": drop, "backend": ch.backend, "backend_status": ch.backend_status})
        _progress(cfg, f"[drop {drop+1}/{num_drops}] channel backend={ch.backend}; computing Gamma measurement")
        meas = compute_gamma_measurement(ch.h_freq, tx_beams, rx_beams, beam_ids,
                                         tx_power_w_per_panel=tx_power_w_per_panel,
                                         noise_power_w=noise_w,
                                         link_adapter=link_adapter)
        ue_site_ids = {int(ue.ue_id): int(ue.site_id) for ue in topo.ues}
        ue_serving_cells = {int(ue.ue_id): int(ue.serving_cell) for ue in topo.ues}
        candidate_beam_indices_by_ue = None
        if is_site_domain_mode(domain_mode):
            beams_by_site = _beam_indices_by_site(beam_ids)
            candidate_beam_indices_by_ue = {
                int(ue.ue_id): beams_by_site.get(int(ue.site_id), [])
                for ue in topo.ues
            }
        reports_by_scheme = make_reports(
            meas, beam_ids, schemes=schemes,
            k1=int(cfg["feedback"].get("service_beam_top_k1", 2)),
            oracle_top_k=int(cfg["feedback"].get("oracle_service_beam_top_k", 4)),
            k2=int(cfg["feedback"].get("conflict_top_k2", 3)),
            threshold_db=float(cfg["feedback"].get("conflict_sinr_threshold_db", 0.0)),
            ue_site_ids=ue_site_ids,
            ue_serving_cells=ue_serving_cells,
            candidate_beam_indices_by_ue=candidate_beam_indices_by_ue,
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

        _progress(cfg, f"[drop {drop+1}/{num_drops}] scheduling {len(schemes)} feedback schemes")
        # Drops are independent UE/topology realizations. Keep OLLA state across
        # TTIs within a drop, but do not leak offsets to the next random drop.
        olla_state: Dict[Tuple[str, int], float] = {}
        for scheme in schemes:
            sched = schedule(reports_by_scheme[scheme], beam_ids, cfg, tbar_by_scheme.get(scheme), link_adapter=link_adapter)
            schedule_rows.append({"drop": drop, "scheme": scheme,
                                  "objective_value": sched.objective_value,
                                  "num_scheduled": len(sched.links),
                                  "schedule_json": json.dumps(sched.to_dict(beam_ids), ensure_ascii=False)})
            scheduler_stat_rows.extend(_schedule_stat_rows(drop, scheme, sched))
            link_rows, olla_state = run_tti_loop(sched, ch.h_freq, tx_beams, rx_beams, beam_ids, meas,
                                                 tx_power_w_per_panel, cfg, drop, rng, olla_state,
                                                 link_adapter=link_adapter)
            all_link_rows.extend(link_rows)
            if cfg["scheduler"].get("objective", "sum_rate") == "proportional_fair":
                alpha = 0.05
                for r in link_rows:
                    old = tbar_by_scheme[scheme].get(r.ue_id, float(cfg["scheduler"].get("pf_tbar_init_mbps", 1.0)))
                    tbar_by_scheme[scheme][r.ue_id] = (1 - alpha) * old + alpha * r.goodput_mbps

        _progress(cfg, f"[drop {drop+1}/{num_drops}] finished")
        drop_rows.append({
            "drop": drop,
            "num_ues": len(topo.ues),
            "num_cells": topo.num_cells,
            "num_sites": len(topo.sites),
            "num_beams": len(beam_ids),
            "scheduler_domain_mode": domain_mode,
            "noise_dbm": float(watt_to_dbm(noise_w)),
            "tx_power_per_tx_unit_dbm": float(watt_to_dbm(tx_power_w_per_panel)),
            "avg_su_snr_db": float(np.mean(meas.su_snr_db)),
            "p95_su_snr_db": float(np.percentile(meas.su_snr_db, 95)),
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
    write_csv(out_dir / "metrics" / "reports.csv", report_rows)
    write_csv(out_dir / "metrics" / "scheduler_stats.csv", scheduler_stat_rows)
    write_csv(out_dir / "metrics" / "channel_backend.csv", channel_backend_rows)

    summary = summarize_results(link_dict_rows, schemes)
    summary["_backend"] = {
        "link_adaptation": link_adapter.status.__dict__,
        "channel_backends": channel_backend_rows,
    }
    write_json(out_dir / "metrics" / "summary.json", summary)
    write_csv(out_dir / "metrics" / "summary.csv", [{"scheme": k, **v} for k, v in summary.items() if isinstance(v, dict) and not k.startswith("_")])

    make_plots(out_dir, link_dict_rows, summary, schemes)

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


def summarize_results(link_rows: List[Dict[str, Any]], schemes: List[str]) -> Dict[str, Any]:
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
        ue_vals: Dict[Tuple[int, int], List[float]] = {}
        for r in rows:
            key = (int(r["drop"]), int(r["tti"]))
            sys_by_slot[key] = sys_by_slot.get(key, 0.0) + float(r["goodput_mbps"])
            ue_key = (int(r["drop"]), int(r["ue_id"]))
            ue_vals.setdefault(ue_key, []).append(float(r["goodput_mbps"]))
        ue_mean = [float(np.mean(v)) for v in ue_vals.values()]
        summary[scheme] = {
            "avg_system_goodput_mbps": float(np.mean(list(sys_by_slot.values()))) if sys_by_slot else 0.0,
            "avg_ue_goodput_mbps": float(np.mean(ue_mean)) if ue_mean else 0.0,
            "p05_ue_goodput_mbps": percentile(ue_mean, 5.0),
            "avg_eff_sinr_db": float(np.mean([float(r["effective_sinr_db"]) for r in rows])),
            "avg_tbler": float(np.mean([float(r["tbler"]) for r in rows])),
            "ack_rate": float(np.mean([float(r["ack"]) for r in rows])),
            "num_tx": len(rows),
        }
    if "full_gamma" in summary:
        oracle = max(float(summary["full_gamma"].get("avg_system_goodput_mbps", 0.0)), 1e-12)
        for scheme in schemes:
            if isinstance(summary.get(scheme), dict):
                summary[scheme]["oracle_ratio"] = float(summary[scheme].get("avg_system_goodput_mbps", 0.0)) / oracle
    if "baseline" in summary:
        base = max(float(summary["baseline"].get("avg_system_goodput_mbps", 0.0)), 1e-12)
        for scheme in schemes:
            if isinstance(summary.get(scheme), dict):
                summary[scheme]["gain_over_baseline"] = (float(summary[scheme].get("avg_system_goodput_mbps", 0.0)) - base) / base
    return summary


def make_plots(out_dir: Path, link_rows: List[Dict[str, Any]], summary: Dict[str, Any], schemes: List[str]) -> None:
    sinr_by_scheme = {s: [float(r["effective_sinr_db"]) for r in link_rows if r.get("scheme") == s] for s in schemes}
    goodput_by_scheme = {s: [float(r["goodput_mbps"]) for r in link_rows if r.get("scheme") == s] for s in schemes}
    plot_cdf(sinr_by_scheme, "Effective SINR [dB]", "Post-SINR CDF", out_dir / "figures" / "effective_sinr_cdf.png")
    plot_cdf(goodput_by_scheme, "TTI link goodput [Mbps]", "Link goodput CDF", out_dir / "figures" / "link_goodput_cdf.png")
    bar = {s: float(summary[s]["avg_system_goodput_mbps"]) for s in schemes if s in summary}
    plot_bar(bar, "Average system goodput [Mbps]", "Average system goodput", out_dir / "figures" / "avg_system_goodput.png")
