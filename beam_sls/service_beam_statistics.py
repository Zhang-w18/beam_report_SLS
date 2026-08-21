from __future__ import annotations

import copy
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping

import numpy as np

from .channel import SionnaImportProbe, generate_channel
from .codebook import (
    ArrayConfig,
    BeamId,
    build_network_tx_beams,
    dft_codebook_from_array,
    extract_panel_tx_dimension,
)
from .config import save_config
from .ftp_traffic import (
    FTPArrivalOnlyConfig,
    count_arriving_ues_by_beam,
    sample_ue_arrivals,
)
from .measurement import associate_ues_by_average_rsrp, compute_gamma_measurement
from .rf import resolve_rf_architecture, trps_per_sector
from .topology import make_topology, topology_to_rows
from .utils import (
    dbm_to_watt,
    ensure_dir,
    occupied_bandwidth_hz,
    percentile,
    thermal_noise_watt,
    write_csv,
    write_json,
)


def _rng_for_drop(seed: int, drop: int, stream: int = 0) -> np.random.Generator:
    """Derive reproducible, independent topology/channel and traffic streams."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(drop), int(stream)]))


def _max_beams_from_cfg(array_section: Mapping[str, Any], array_cfg: ArrayConfig) -> int | None:
    value = array_section.get("max_beams")
    if value is not None:
        return int(value)
    configured = array_cfg.configured_beams_per_codebook
    return None if configured is None else int(configured)


def _panels_per_cell(cfg: Mapping[str, Any], rf_architecture) -> int:
    return max(1, trps_per_sector(cfg)) * max(1, len(rf_architecture.tx_units))


def _resolve_tx_power_w_per_panel(cfg: Mapping[str, Any], tx_cfg: ArrayConfig) -> float:
    return float(dbm_to_watt(float(cfg["system"]["tx_power_dbm"]))) / max(
        1, int(tx_cfg.num_array_panels)
    )


def _beam_meta(beam_ids: List[BeamId]) -> Dict[int, Dict[str, Any]]:
    return {
        int(index): {
            "beam_index": int(index),
            "beam_id": beam.short(),
            "cell": int(beam.cell),
            "trp": int(beam.trp),
            "panel": int(beam.panel),
            "local_beam": int(beam.beam),
        }
        for index, beam in enumerate(beam_ids)
    }


def select_best_service_beam(service_power_w: np.ndarray, candidate_indices: List[int]) -> int:
    """Select maximum received-power TX beam with deterministic global tie-break."""
    if not candidate_indices:
        raise ValueError("UE has no candidate service beam")
    return min(
        (int(index) for index in candidate_indices),
        key=lambda index: (-float(service_power_w[index]), index),
    )


def _pmf_rows(
    beam_ids: List[BeamId],
    samples_by_beam: Mapping[int, List[int]],
    traffic: FTPArrivalOnlyConfig,
    num_observations: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta = _beam_meta(beam_ids)
    for beam_index, info in meta.items():
        samples = np.asarray(samples_by_beam.get(beam_index, []), dtype=int)
        if samples.size != int(num_observations):
            raise ValueError(
                f"beam {beam_index} has {samples.size} samples, expected {num_observations}"
            )
        busy_count = int(np.count_nonzero(samples > 0))
        for ue_count in range(int(samples.max()) + 1 if samples.size else 1):
            sample_count = int(np.count_nonzero(samples == ue_count))
            rows.append({
                **info,
                "ue_count": int(ue_count),
                "sample_count": sample_count,
                "probability": float(sample_count / num_observations),
                "conditional_probability_given_nonzero": (
                    float(sample_count / busy_count) if busy_count and ue_count > 0 else 0.0
                ),
                "arrival_rate_per_ue_s": float(traffic.arrival_rate_per_ue_s),
                "observation_interval_ms": float(traffic.observation_interval_s * 1e3),
                "file_size_mbytes": float(traffic.file_size_mbytes),
            })
    return rows


def _summary_rows(
    beam_ids: List[BeamId],
    samples_by_beam: Mapping[int, List[int]],
    traffic: FTPArrivalOnlyConfig,
    num_observations: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta = _beam_meta(beam_ids)
    for beam_index, info in meta.items():
        samples = np.asarray(samples_by_beam.get(beam_index, []), dtype=float)
        busy_probability = float(np.mean(samples > 0)) if samples.size else 0.0
        rows.append({
            **info,
            "num_observations": int(num_observations),
            "mean_ue_count": float(np.mean(samples)) if samples.size else 0.0,
            "variance_ue_count": float(np.var(samples)) if samples.size else 0.0,
            "idle_probability": float(1.0 - busy_probability),
            "busy_probability": busy_probability,
            "p50_ue_count": percentile(samples, 50.0),
            "p90_ue_count": percentile(samples, 90.0),
            "p95_ue_count": percentile(samples, 95.0),
            "arrival_rate_per_ue_s": float(traffic.arrival_rate_per_ue_s),
            "observation_interval_ms": float(traffic.observation_interval_s * 1e3),
            "file_size_mbytes": float(traffic.file_size_mbytes),
        })
    return rows


def _selection_rows(
    beam_ids: List[BeamId],
    selection_counts: Mapping[int, int],
    candidate_counts_by_cell: Mapping[int, int],
    drop: int,
) -> List[Dict[str, Any]]:
    rows = []
    for beam_index, info in _beam_meta(beam_ids).items():
        cell_count = int(candidate_counts_by_cell.get(info["cell"], 0))
        selected = int(selection_counts.get(beam_index, 0))
        rows.append({
            "drop": int(drop),
            **info,
            "candidate_ue_count": selected,
            "candidate_cell_ue_count": cell_count,
            "selection_probability": float(selected / cell_count) if cell_count else 0.0,
        })
    return rows


def run_service_beam_statistics(cfg: Dict[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    """Run fixed-UE FTP arrival-only service-beam statistics.

    This function intentionally has no feedback, evaluation, scheduler, link
    adaptation, OLLA, ACK/NACK, TBLER, throughput, or TTI-loop calls.
    """
    out_dir = ensure_dir(out_dir)
    stats_cfg = cfg.get("service_beam_statistics", {}) or {}
    traffic_cfg = stats_cfg.get("traffic", {}) or {}
    model = str(traffic_cfg.get("model", "3gpp_ftp_model_3_arrival_only")).lower()
    if model not in {
        "3gpp_ftp_model_3_arrival_only",
        "ftp_model_3_arrival_only",
        "ftp3_arrival_only",
    }:
        raise ValueError(
            "service_beam_statistics.traffic.model must be "
            "3gpp_ftp_model_3_arrival_only"
        )
    traffic = FTPArrivalOnlyConfig.from_dict({**stats_cfg, **traffic_cfg})
    num_observations = int(stats_cfg.get("observations_per_drop", 1))
    warmup_observations = int(stats_cfg.get("warmup_observations", 0))
    if num_observations <= 0:
        raise ValueError("service_beam_statistics.observations_per_drop must be > 0")
    if warmup_observations < 0:
        raise ValueError("service_beam_statistics.warmup_observations must be >= 0")
    if not bool(stats_cfg.get("include_zero_count", True)):
        raise ValueError("service_beam_statistics.include_zero_count must be true")

    effective_cfg = copy.deepcopy(cfg)
    candidate_ues = stats_cfg.get("candidate_ues_per_sector_per_drop")
    if candidate_ues is not None:
        candidate_ues = int(candidate_ues)
        if candidate_ues <= 0:
            raise ValueError(
                "service_beam_statistics.candidate_ues_per_sector_per_drop must be > 0"
            )
        effective_cfg.setdefault("ue_drop", {})["num_ut_per_sector"] = candidate_ues
    candidate_ues_per_sector = int(effective_cfg["ue_drop"]["num_ut_per_sector"])
    random_seed = int(effective_cfg["system"].get("random_seed", 1))
    num_drops = int(effective_cfg["system"].get("num_drops", 1))
    if num_drops <= 0:
        raise ValueError("system.num_drops must be > 0")

    cfg.setdefault("_resolved", {}).update({
        "run_mode": "service_beam_statistics",
        "ftp_model": "3gpp_ftp_model_3_arrival_only",
        "candidate_ues_per_sector_per_drop": candidate_ues_per_sector,
        "observations_per_drop": num_observations,
        "warmup_observations": warmup_observations,
    })
    save_config(out_dir / "resolved_config.yaml", cfg)

    tx_cfg = ArrayConfig.from_dict(effective_cfg["tx_array"])
    rx_cfg = ArrayConfig.from_dict(effective_cfg["ue_array"])
    rf_arch = resolve_rf_architecture(effective_cfg, tx_cfg)
    tx_power_w_per_panel = _resolve_tx_power_w_per_panel(effective_cfg, tx_cfg)
    topo0 = make_topology(effective_cfg, _rng_for_drop(random_seed, 0, 999))
    site_id_by_cell = [topo0.sector_by_cell(c).site_id for c in range(topo0.num_cells)]
    beam_ids, tx_beams = build_network_tx_beams(
        num_cells=topo0.num_cells,
        panels_per_cell=_panels_per_cell(effective_cfg, rf_arch),
        tx_cfg=tx_cfg,
        max_beams_per_panel=_max_beams_from_cfg(effective_cfg["tx_array"], tx_cfg),
        site_id_by_cell=site_id_by_cell,
        rf_architecture=rf_arch,
    )
    rx_beams = dft_codebook_from_array(
        rx_cfg, max_beams=_max_beams_from_cfg(effective_cfg["ue_array"], rx_cfg)
    )
    beam_meta = _beam_meta(beam_ids)
    beams_by_cell: Dict[int, List[int]] = {}
    for beam_index, info in beam_meta.items():
        beams_by_cell.setdefault(int(info["cell"]), []).append(beam_index)
    noise_w = thermal_noise_watt(
        occupied_bandwidth_hz(effective_cfg),
        noise_density_dbm_per_hz=float(
            effective_cfg["noise"].get("thermal_noise_density_dbm_per_hz", -174.0)
        ),
        noise_figure_db=float(effective_cfg["noise"].get("ue_noise_figure_db", 7.0)),
    )
    if effective_cfg.get("sionna", {}).get("enable_import_probe", True):
        write_json(out_dir / "sionna_import_probe.json", SionnaImportProbe().run())
    write_json(out_dir / "rf_architecture_summary.json", rf_arch.to_dict())
    write_json(out_dir / "array_config_summary.json", {
        "tx_array": tx_cfg.to_dict(),
        "ue_array": rx_cfg.to_dict(),
        "tx_beams_per_codebook": _max_beams_from_cfg(effective_cfg["tx_array"], tx_cfg),
        "ue_rx_beams": _max_beams_from_cfg(effective_cfg["ue_array"], rx_cfg),
        "rf_architecture": rf_arch.to_dict(),
        "tx_power_per_panel_dbm": float(10.0 * np.log10(max(tx_power_w_per_panel, 1e-30)) + 30.0),
    })

    cache_rows: List[Dict[str, Any]] = []
    arrival_rows: List[Dict[str, Any]] = []
    count_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    site_rows_all: List[Dict[str, Any]] = []
    sector_rows_all: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    drop_rows: List[Dict[str, Any]] = []
    all_samples: Dict[int, List[int]] = {int(i): [] for i in range(len(beam_ids))}
    save_arrival_samples = bool(stats_cfg.get("save_per_ue_arrival_samples", False))

    for drop in range(num_drops):
        topology_rng = _rng_for_drop(random_seed, drop, 0)
        traffic_rng = _rng_for_drop(random_seed, drop, 1)
        phase_start = perf_counter()
        topo = make_topology(effective_cfg, topology_rng)
        site_rows, sector_rows = topology_to_rows(topo)
        site_rows_all.extend({"drop": drop, **row} for row in site_rows)
        sector_rows_all.extend({"drop": drop, **row} for row in sector_rows)
        channel_start = perf_counter()
        ch = generate_channel(topo, effective_cfg, tx_cfg, rx_cfg, topology_rng)
        channel_rows.append({"drop": drop, "backend": ch.backend, "backend_status": ch.backend_status})
        runtime_rows.append({
            "drop": drop,
            "phase": "topology_and_channel",
            "elapsed_s": float(perf_counter() - phase_start),
            "channel_backend": ch.backend,
        })
        runtime_rows.append({
            "drop": drop,
            "phase": "channel_generation",
            "elapsed_s": float(perf_counter() - channel_start),
            "channel_backend": ch.backend,
        })
        measurement_h = (
            extract_panel_tx_dimension(ch.h_freq, tx_cfg, int(rf_arch.measurement_panel_index))
            if rf_arch.compact_panel_channel
            else ch.h_freq
        )
        associate_start = perf_counter()
        associate_ues_by_average_rsrp(
            measurement_h, tx_beams, rx_beams, beam_ids, topo, tx_power_w_per_panel
        )
        runtime_rows.append({
            "drop": drop,
            "phase": "average_rsrp_association",
            "elapsed_s": float(perf_counter() - associate_start),
        })
        service_indices_by_ue = {
            int(ue.ue_id): list(beams_by_cell.get(int(ue.serving_cell), []))
            for ue in topo.ues
        }
        measurement_start = perf_counter()
        measurement = compute_gamma_measurement(
            measurement_h,
            tx_beams,
            rx_beams,
            beam_ids,
            tx_power_w_per_panel=tx_power_w_per_panel,
            noise_power_w=noise_w,
            service_beam_indices_by_ue=service_indices_by_ue,
            interference_beam_indices_by_ue=service_indices_by_ue,
            compute_backend=effective_cfg["measurement"].get("gamma_backend", "numpy"),
            ue_batch_size=effective_cfg["measurement"].get("gamma_ue_batch_size", 0),
        )
        runtime_rows.append({
            "drop": drop,
            "phase": "service_beam_measurement",
            "elapsed_s": float(perf_counter() - measurement_start),
            "backend": measurement.compute_backend,
        })

        best_service_beam_by_ue: Dict[int, int] = {}
        selection_counts = {int(index): 0 for index in range(len(beam_ids))}
        candidate_counts_by_cell: Dict[int, int] = {}
        for ue_index, ue in enumerate(topo.ues):
            candidates = service_indices_by_ue[int(ue.ue_id)]
            best = select_best_service_beam(measurement.service_power_w[ue_index], candidates)
            best_service_beam_by_ue[int(ue.ue_id)] = best
            selection_counts[best] += 1
            candidate_counts_by_cell[int(ue.serving_cell)] = (
                candidate_counts_by_cell.get(int(ue.serving_cell), 0) + 1
            )
            cache_rows.append({
                "drop": int(drop),
                "ue_id": int(ue.ue_id),
                "x_m": float(ue.x_m),
                "y_m": float(ue.y_m),
                "serving_cell": int(ue.serving_cell),
                "best_service_beam_index": int(best),
                "best_service_beam_id": beam_ids[best].short(),
                "best_rx_beam_index": int(measurement.selected_rx_beam[ue_index, best]),
                "service_power_w": float(measurement.service_power_w[ue_index, best]),
            })
        selection_rows.extend(
            _selection_rows(beam_ids, selection_counts, candidate_counts_by_cell, drop)
        )

        for observation_with_warmup in range(warmup_observations + num_observations):
            num_arrivals = sample_ue_arrivals(traffic_rng, traffic, len(topo.ues))
            if observation_with_warmup < warmup_observations:
                continue
            observation = observation_with_warmup - warmup_observations
            if save_arrival_samples:
                for ue_index, ue in enumerate(topo.ues):
                    beam_index = best_service_beam_by_ue[int(ue.ue_id)]
                    arrival_rows.append({
                        "drop": int(drop),
                        "observation": int(observation),
                        "ue_id": int(ue.ue_id),
                        "num_arrivals": int(num_arrivals[ue_index]),
                        "has_arrival": int(num_arrivals[ue_index] > 0),
                        "serving_cell": int(ue.serving_cell),
                        "best_service_beam_index": int(beam_index),
                        "best_service_beam_id": beam_ids[beam_index].short(),
                    })
            else:
                for ue_index, ue in enumerate(topo.ues):
                    if int(num_arrivals[ue_index]) <= 0:
                        continue
                    beam_index = best_service_beam_by_ue[int(ue.ue_id)]
                    arrival_rows.append({
                        "drop": int(drop),
                        "observation": int(observation),
                        "ue_id": int(ue.ue_id),
                        "num_arrivals": int(num_arrivals[ue_index]),
                        "has_arrival": 1,
                        "serving_cell": int(ue.serving_cell),
                        "best_service_beam_index": int(beam_index),
                        "best_service_beam_id": beam_ids[beam_index].short(),
                    })
            counts_by_beam = count_arriving_ues_by_beam(
                [ue.ue_id for ue in topo.ues],
                num_arrivals,
                best_service_beam_by_ue,
                len(beam_ids),
            )
            for beam_index, count in enumerate(counts_by_beam):
                all_samples[beam_index].append(int(count))
                count_rows.append({
                    "drop": int(drop),
                    "observation": int(observation),
                    "beam_index": int(beam_index),
                    "beam_id": beam_meta[beam_index]["beam_id"],
                    "cell": beam_meta[beam_index]["cell"],
                    "ue_with_arrival_count": int(count),
                })
        drop_rows.append({
            "drop": int(drop),
            "num_cells": int(topo.num_cells),
            "num_candidate_ues": int(len(topo.ues)),
            "num_beams": int(len(beam_ids)),
            "num_observations": int(num_observations),
            "channel_backend": ch.backend,
            "channel_backend_status": ch.backend_status,
            "traffic_model": "3gpp_ftp_model_3_arrival_only",
        })

    total_observations = int(num_drops * num_observations)
    pmf_rows = _pmf_rows(beam_ids, all_samples, traffic, total_observations)
    summary_rows = _summary_rows(beam_ids, all_samples, traffic, total_observations)
    metrics_dir = out_dir / "metrics"
    write_csv(metrics_dir / "service_beam_ue_cache.csv", cache_rows)
    write_csv(metrics_dir / "ftp_ue_arrival_samples.csv", arrival_rows)
    write_csv(metrics_dir / "service_beam_ue_count_samples.csv", count_rows)
    write_csv(metrics_dir / "service_beam_ue_count_pmf.csv", pmf_rows)
    write_csv(metrics_dir / "service_beam_ue_count_summary.csv", summary_rows)
    write_csv(metrics_dir / "service_beam_selection_probability.csv", selection_rows)
    write_csv(metrics_dir / "beams.csv", [beam.to_dict() for beam in beam_ids])
    write_csv(metrics_dir / "sites.csv", site_rows_all)
    write_csv(metrics_dir / "sectors.csv", sector_rows_all)
    write_csv(metrics_dir / "drops.csv", drop_rows)
    write_csv(metrics_dir / "channel_backend.csv", channel_rows)
    write_csv(metrics_dir / "runtime_phases.csv", runtime_rows)
    summary = {
        "mode": "service_beam_statistics",
        "ftp_model": "3gpp_ftp_model_3_arrival_only",
        "num_drops": num_drops,
        "observations_per_drop": num_observations,
        "warmup_observations": warmup_observations,
        "num_beams": len(beam_ids),
        "num_candidate_ues_per_drop": candidate_ues_per_sector * topo0.num_cells,
        "arrival_rate_per_ue_s": traffic.arrival_rate_per_ue_s,
        "observation_interval_ms": traffic.observation_interval_s * 1e3,
        "file_size_mbytes": traffic.file_size_mbytes,
        "mean_beam_ue_count": float(np.mean([row["mean_ue_count"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_beam_busy_probability": float(np.mean([row["busy_probability"] for row in summary_rows])) if summary_rows else 0.0,
        "outputs": {
            "ue_cache": "metrics/service_beam_ue_cache.csv",
            "arrival_samples": "metrics/ftp_ue_arrival_samples.csv",
            "ue_count_samples": "metrics/service_beam_ue_count_samples.csv",
            "pmf": "metrics/service_beam_ue_count_pmf.csv",
            "summary": "metrics/service_beam_ue_count_summary.csv",
        },
    }
    write_json(metrics_dir / "summary.json", summary)
    write_csv(metrics_dir / "summary.csv", [summary])
    return summary
