from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .channel import generate_channel, generate_numpy_geometric_channel, uma_like_pathloss_db
from .codebook import ArrayConfig, BeamId, build_network_tx_beams, steering_vector_from_array
from .measurement import compute_gamma_measurement
from .topology import Sector, Site, Topology, UE
from .utils import db_to_lin, watt_to_dbm


def compute_coverage_heatmap_los_rank1(cfg: Dict,
                                        tx_array: ArrayConfig,
                                        rx_array: ArrayConfig,
                                        tx_beam_codebook_single_panel: np.ndarray,
                                        rx_beams: np.ndarray,
                                        tx_power_w_per_panel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v1 fallback: rank-1 LOS heatmap."""
    hm = cfg.get("coverage_heatmap", {})
    grid_size = int(hm.get("grid_size", 50))
    max_d = float(hm.get("max_distance_m", 300.0))
    fc = float(cfg["scenario"]["carrier_frequency_ghz"])
    exponent = float(cfg["scenario"].get("pathloss_exponent", 3.0))
    sector_az = np.deg2rad(float(cfg["scenario"].get("sector_azimuth_deg", 30.0)))
    panel_offsets = cfg["trp"].get("panel_azimuth_offsets_deg", [0.0])
    x = np.linspace(-max_d, max_d, grid_size)
    y = np.linspace(-max_d, max_d, grid_size)
    p_dbm = np.full((grid_size, grid_size), np.nan, dtype=float)

    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            d = float(np.hypot(xx, yy))
            if d < 1.0 or d > max_d:
                continue
            az = float(np.arctan2(yy, xx))
            pl = uma_like_pathloss_db(d, fc, exponent)
            gain = float(db_to_lin(-pl))
            panel_vals = []
            for off in panel_offsets:
                rel = az - (sector_az + np.deg2rad(float(off)))
                rel = float(np.arctan2(np.sin(rel), np.cos(rel)))
                atx = steering_vector_from_array(tx_array, rel, 0.0)
                arx = steering_vector_from_array(rx_array, np.pi + rel, 0.0)
                vals = []
                for f in tx_beam_codebook_single_panel:
                    h_f = np.sqrt(gain) * arx * np.vdot(atx, f)
                    z = np.conjugate(rx_beams) @ h_f
                    vals.append(tx_power_w_per_panel * float(np.max(np.abs(z) ** 2)))
                panel_vals.append(float(np.mean(vals)))
            p_dbm[iy, ix] = float(watt_to_dbm(np.mean(panel_vals)))
    return x, y, p_dbm


def _make_grid_points(cfg: Dict) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float, float]]]:
    hm = cfg.get("coverage_heatmap", {})
    grid_size = int(hm.get("grid_size", 40))
    max_d = float(hm.get("max_distance_m", 300.0))
    min_d = float(cfg.get("scenario", {}).get("min_ue_distance_m", 35.0))
    x = np.linspace(-max_d, max_d, grid_size)
    y = np.linspace(-max_d, max_d, grid_size)
    pts: List[Tuple[int, int, float, float]] = []
    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            d = float(np.hypot(xx, yy))
            if d < min_d or d > max_d:
                continue
            pts.append((iy, ix, float(xx), float(yy)))
    return x, y, pts


def _topology_for_grid_chunk(cfg: Dict, pts: List[Tuple[int, int, float, float]], base_topology: Topology | None = None) -> Topology:
    sc = cfg["scenario"]
    topo_cfg = cfg.get("topology", {})
    isd = float(topo_cfg.get("isd_m", sc.get("isd_m", 500.0)))
    bs_height = float(topo_cfg.get("bs_height_m", 25.0))
    sites = [Site(0, 0.0, 0.0, bs_height)]
    azs = topo_cfg.get("sector_azimuths_deg", [30.0, 150.0, 270.0])
    sectors = [Sector(cell_id=i, site_id=0, sector_id=i, azimuth_deg=float(a),
                      width_deg=float(topo_cfg.get("sector_width_deg", sc.get("sector_width_deg", 120.0))))
               for i, a in enumerate(azs)]
    ues = []
    for uid, (_, _, xx, yy) in enumerate(pts):
        # Assign serving cell by closest boresight. Heatmap itself uses best beam
        # over all cells/sectors, so this label is diagnostic only.
        az = np.rad2deg(np.arctan2(yy, xx))
        diffs = [abs(float(np.rad2deg(np.arctan2(np.sin(np.deg2rad(az-a)), np.cos(np.deg2rad(az-a)))))) for a in azs]
        cell = int(np.argmin(diffs)) if diffs else 0
        ues.append(UE(uid, xx, yy, z_m=float(topo_cfg.get("ue_height_m", 1.5)), serving_cell=cell, site_id=0))
    return Topology(ues=ues, sites=sites, sectors=sectors,
                    carrier_frequency_ghz=float(sc["carrier_frequency_ghz"]),
                    isd_m=isd, layout="coverage_grid",
                    sector_azimuth_deg=float(azs[0]) if azs else 0.0,
                    sector_width_deg=float(topo_cfg.get("sector_width_deg", sc.get("sector_width_deg", 120.0))))


def compute_coverage_heatmap_standard_sampling(cfg: Dict,
                                               tx_array: ArrayConfig,
                                               rx_array: ArrayConfig,
                                               tx_beams: np.ndarray,
                                               rx_beams: np.ndarray,
                                               beam_ids: List[BeamId],
                                               tx_power_w_per_panel: float,
                                               noise_power_w: float,
                                               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Coverage map by sampling the configured channel backend at each grid point.

    For sionna_tr38901_uma/umi/rma, this calls the Sionna backend through
    generate_channel(). If Sionna is unavailable and fallback is enabled, the
    backend status records the fallback and the same sampling loop is executed
    with the v1 geometric channel.
    """
    hm = cfg.get("coverage_heatmap", {})
    chunk = int(hm.get("chunk_size", 128))
    x, y, pts = _make_grid_points(cfg)
    p_dbm = np.full((len(y), len(x)), np.nan, dtype=float)
    best_beam = np.full((len(y), len(x)), -1, dtype=int)
    backend_status = "not_run"
    for start in range(0, len(pts), max(1, chunk)):
        part = pts[start:start + max(1, chunk)]
        topo = _topology_for_grid_chunk(cfg, part)
        ch = generate_channel(topo, cfg, tx_array, rx_array, rng)
        backend_status = ch.backend_status
        meas = compute_gamma_measurement(ch.h_freq, tx_beams, rx_beams, beam_ids,
                                         tx_power_w_per_panel=tx_power_w_per_panel,
                                         noise_power_w=noise_power_w,
                                         link_adapter=None,
                                         compute_backend=cfg.get("measurement", {}).get("gamma_backend", "numpy"),
                                         ue_batch_size=cfg.get("measurement", {}).get("gamma_ue_batch_size", 0))
        # Best service beam RSRP-like power, independent of scheduler.
        vals = np.max(meas.service_power_w, axis=1)
        bidx = np.argmax(meas.service_power_w, axis=1)
        for local_i, (iy, ix, _, _) in enumerate(part):
            p_dbm[iy, ix] = float(watt_to_dbm(vals[local_i]))
            best_beam[iy, ix] = int(bidx[local_i])
    return x, y, p_dbm, best_beam, backend_status


def compute_coverage_heatmap(cfg: Dict,
                             tx_array: ArrayConfig,
                             rx_array: ArrayConfig,
                             tx_beam_codebook_single_panel: np.ndarray,
                             rx_beams: np.ndarray,
                             tx_power_w_per_panel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible v1 API."""
    return compute_coverage_heatmap_los_rank1(cfg, tx_array, rx_array,
                                              tx_beam_codebook_single_panel, rx_beams,
                                              tx_power_w_per_panel)


def _candidate_vertical_indices(cfg: Dict, tx_array: ArrayConfig) -> List[int]:
    fvb = cfg.get("coverage_heatmap", {}).get("fixed_vertical_beam_cdf", {})
    candidates = fvb.get("candidate_v_indices", "all")
    n_v = tx_array.panel_num_v if tx_array.normalized_beam_scope == "per_panel" else tx_array.num_v
    if candidates == "all" or candidates is None:
        return list(range(int(n_v)))
    return [int(v) % int(n_v) for v in candidates]


def _metric_from_values(vals: np.ndarray, metric: str) -> float:
    metric = str(metric or "mean_dbm").lower()
    if vals.size == 0:
        return float("-inf")
    if metric in ("p05_dbm", "p5_dbm", "cell_edge_dbm"):
        return float(np.percentile(vals, 5))
    if metric in ("p50_dbm", "median_dbm"):
        return float(np.percentile(vals, 50))
    if metric in ("p95_dbm",):
        return float(np.percentile(vals, 95))
    return float(np.mean(vals))


def compute_fixed_vertical_beam_cdf(cfg: Dict,
                                    tx_array: ArrayConfig,
                                    rx_array: ArrayConfig,
                                    rx_beams: np.ndarray,
                                    site_id_by_cell: List[int],
                                    tx_power_w_per_panel: float,
                                    noise_power_w: float,
                                    rng: np.random.Generator,
                                    panels_per_cell: int,
                                    rf_architecture=None) -> Tuple[List[Dict], List[Dict], int]:
    """Evaluate fixed vertical DFT beams as electrical downtilt candidates.

    For each candidate vertical DFT index v, we fix v and scan horizontal DFT
    beams. At every grid point, RSRP is averaged over all horizontal scanning
    beams for each sector/panel group, and then the best sector/panel group is
    used as the point's coverage RSRP. The CDF across grid points is used to
    select a vertical beam according to selection_metric.
    """
    fvb = cfg.get("coverage_heatmap", {}).get("fixed_vertical_beam_cdf", {})
    chunk = int(cfg.get("coverage_heatmap", {}).get("chunk_size", 128))
    metric = str(fvb.get("selection_metric", "mean_dbm"))
    original_h = tx_array.num_beams_h
    h_num = int(fvb.get("horizontal_num_beams", original_h if original_h is not None else tx_array.codebook_num_h))
    # Keep the array immutable: create a shallow dataclass replacement manually.
    tx_scan = ArrayConfig(
        num_h=tx_array.num_h, num_v=tx_array.num_v,
        d_h_lambda=tx_array.d_h_lambda, d_v_lambda=tx_array.d_v_lambda,
        polarization_count=tx_array.polarization_count, num_txru=tx_array.num_txru,
        num_beams_h=h_num, num_beams_v=1,
        beam_scope=tx_array.beam_scope, sampling_mode=tx_array.sampling_mode,
        vertical_beam_mode="fixed", fixed_v_index=None,
        model=tx_array.model, M=tx_array.M, N=tx_array.N, P=tx_array.P,
        Mg=tx_array.Mg, Ng=tx_array.Ng, Mp=tx_array.Mp, Np=tx_array.Np,
    )
    candidates = _candidate_vertical_indices(cfg, tx_array)
    x, y, pts = _make_grid_points(cfg)
    values_by_v: Dict[int, List[float]] = {int(v): [] for v in candidates}

    for start in range(0, len(pts), max(1, chunk)):
        part = pts[start:start + max(1, chunk)]
        topo = _topology_for_grid_chunk(cfg, part)
        ch = generate_channel(topo, cfg, tx_array, rx_array, rng)
        for v in candidates:
            beam_ids_v, tx_beams_v = build_network_tx_beams(
                num_cells=topo.num_cells,
                panels_per_cell=panels_per_cell,
                tx_cfg=tx_scan,
                max_beams_per_panel=h_num,
                site_id_by_cell=site_id_by_cell,
                fixed_v_index=int(v),
                rf_architecture=rf_architecture,
            )
            meas = compute_gamma_measurement(ch.h_freq, tx_beams_v, rx_beams, beam_ids_v,
                                             tx_power_w_per_panel=tx_power_w_per_panel,
                                             noise_power_w=noise_power_w,
                                             link_adapter=None,
                                             compute_backend=cfg.get("measurement", {}).get("gamma_backend", "numpy"),
                                             ue_batch_size=cfg.get("measurement", {}).get("gamma_ue_batch_size", 0))
            # Average over horizontal scan beams within each sector/panel; then
            # choose the best sector/panel group for the grid point.
            group_to_indices: Dict[Tuple[int, int], List[int]] = {}
            for bi, bid in enumerate(beam_ids_v):
                group_to_indices.setdefault((bid.cell, bid.panel), []).append(bi)
            group_vals = []
            for inds in group_to_indices.values():
                group_vals.append(np.mean(meas.service_power_w[:, inds], axis=1))
            if group_vals:
                vals_w = np.max(np.stack(group_vals, axis=1), axis=1)
            else:
                vals_w = np.zeros((len(part),), dtype=float)
            values_by_v[int(v)].extend([float(watt_to_dbm(x)) for x in vals_w])

    summary: List[Dict] = []
    samples: List[Dict] = []
    best_v = int(candidates[0]) if candidates else 0
    best_score = float("-inf")
    for v in candidates:
        vals = np.asarray(values_by_v[int(v)], dtype=float)
        score = _metric_from_values(vals, metric)
        if score > best_score:
            best_score = score
            best_v = int(v)
        summary.append({
            "v_index": int(v),
            "label": f"v={int(v)} {metric}={score:.2f} dBm",
            "num_points": int(vals.size),
            "selection_metric": metric,
            "selection_score_dbm": float(score),
            "mean_dbm": float(np.mean(vals)) if vals.size else float("nan"),
            "p05_dbm": float(np.percentile(vals, 5)) if vals.size else float("nan"),
            "p50_dbm": float(np.percentile(vals, 50)) if vals.size else float("nan"),
            "p95_dbm": float(np.percentile(vals, 95)) if vals.size else float("nan"),
            "values_dbm": [float(x) for x in vals],
        })
        for i, val in enumerate(vals):
            samples.append({"v_index": int(v), "point_index": int(i), "coverage_rsrp_avg_h_dbm": float(val)})
    return summary, samples, best_v
