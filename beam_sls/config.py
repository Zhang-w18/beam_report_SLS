from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "scenario": {
        "name": "urban_macro_v2_one_site_three_sector",
        "carrier_frequency_ghz": 30.0,
        "isd_m": 500.0,
        "sector_azimuth_deg": 30.0,
        "sector_width_deg": 120.0,
        "min_ue_distance_m": 35.0,
        "max_ue_distance_m": 250.0,
        # v2 supports: numpy_geometric_uma, sionna_tr38901_uma,
        # sionna_tr38901_umi, sionna_tr38901_rma.
        "channel_model": "sionna_tr38901_uma",
        # Fallback channel parameters used only if Sionna backend is unavailable
        # and sionna.fallback_to_numpy_if_unavailable=true.
        "num_clusters": 8,
        "delay_spread_ns": 100.0,
        "shadow_fading_std_db": 4.0,
        "pathloss_exponent": 3.0,
        "enable_pathloss": True,
        "enable_shadow_fading": True,
        "o2i_model": "low",
    },
    "topology": {
        "layout": "one_site_three_sector",
        "num_sites": 1,
        "sectors_per_site": 3,
        "sector_azimuths_deg": [30.0, 150.0, 270.0],
        "sector_width_deg": 120.0,
        "isd_m": 500.0,
        "bs_height_m": 25.0,
    },
    "system": {
        "bandwidth_mhz": 20.0,
        "subcarrier_spacing_khz": 120.0,
        "tx_power_dbm": 33.0,
        "num_drops": 10,
        "num_tti_per_drop": 50,
        "random_seed": 20260610,
        "target_bler": 0.1,
    },
    "pdsch": {
        "num_prbs": 132,
        "num_symbols": 12,
        "dmrs_overhead_re_per_prb": 18,
        "num_layers_per_ue": 1,
        "slot_duration_ms": 0.125,
    },
    "noise": {
        "thermal_noise_density_dbm_per_hz": -174.0,
        "ue_noise_figure_db": 7.0,
    },
    "ue_drop": {
        "num_ut_per_sector": 10,
        "distribution": "uniform_in_sector",
        "speed_kmh": 3.0,
    },
    "trp": {
        # v2.4: one TRP per sector by default. RF architecture determines how
        # many independently schedulable TX units/beams each TRP exposes.
        "num_trps_per_sector": 1,
        # Backward-compatible alias; kept equal to num_trps_per_sector.
        "num_panels_per_sector": 1,
        "panel_azimuth_offsets_deg": [0.0, 0.0],
        "panel_power_mode": "per_tx_unit_equal",
    },
    "rf_architecture": {
        # case 1: panel_polarization_subarray/sub-connected;
        # case 2: fully_connected hybrid beamforming.
        "txru_connectivity": "panel_polarization_subarray",
        "allow_independent_polarization_beams": True,
        "num_txru": 4,
        # auto means scheduler.max_mu_order is derived from the RF architecture.
        "max_parallel_beams_per_trp": "auto",
    },
    "tx_array": {
        # 3GPP-style TRP antenna notation. Default is the requested TRP:
        # 4 TXRUs, 1024 AEs, (M,N,P,Mg,Ng;Mp,Np)=(16,16,2,2,1;1,1).
        "model": "tr38901_panel",
        "num_txru": 4,
        "num_ae": 1024,
        "M": 16,
        "N": 16,
        "P": 2,
        "Mg": 2,
        "Ng": 1,
        "Mp": 1,
        "Np": 1,
        "dH": 0.5,
        "dV": 0.5,
        # SLS candidate beam grid for each independently schedulable sector-panel.
        # The full spatial DFT grid would be num_h*num_v=16*32=512 beams;
        # default SLS uses 4*4=16 candidate beams over the full 1024-AE array.
        # tx_array.beam_scope is retained as a manual override/legacy field.
        # With rf_architecture enabled, the effective beam scope is resolved from
        # rf_architecture.txru_connectivity.
        "beam_scope": "per_panel",
        # Codebook size is N*Ng*Np*M*Mg*Mp. num_beams_h*num_beams_v is the
        # uniformly sampled SLS subset. In per_panel mode these counts apply
        # to each physical panel; in joint mode they apply to the whole TRP.
        "sampling_mode": "uniform",
        "num_beams_h": 4,
        "num_beams_v": 4,
        "max_beams": 16,
        # vertical_beam_mode=scan scans sampled vertical beams. Set to fixed
        # and fixed_v_index=<DFT vertical index> to use an electrical downtilt.
        "vertical_beam_mode": "scan",
        "fixed_v_index": None,
    },
    "ue_array": {
        # UE array is now parameterized with the same 3GPP-style notation.
        # Default keeps the previous 4x4 single-polarized UE, i.e. 16 AEs.
        "model": "tr38901_panel",
        "num_rxru": 4,
        "num_ae": 16,
        "M": 4,
        "N": 4,
        "P": 1,
        "Mg": 1,
        "Ng": 1,
        "Mp": 1,
        "Np": 1,
        "dH": 0.5,
        "dV": 0.5,
        "beam_scope": "joint",
        "sampling_mode": "uniform",
        "num_beams_h": 4,
        "num_beams_v": 4,
        "max_beams": 16,
        "vertical_beam_mode": "scan",
        "fixed_v_index": None,
    },
    "beam": {
        "tx_codebook": "dft_2d",
        "rx_codebook": "dft_2d",
        "one_beam_per_panel": True,
    },
    "measurement": {
        "num_freq_points": 24,
        "compute_full_gamma": True,
        "frequency_average": "linear_power",
    },
    "feedback": {
        "schemes": ["full_gamma", "baseline", "topk_conflict_id", "threshold_conflict_set"],
        # v2: CQI/MCS selection uses link_abstraction.mode. If Sionna SYS is
        # available, this becomes Sionna InnerLoopLinkAdaptation at target BLER.
        "cqi_mode": "illa_target_bler",
        "service_beam_top_k1": 2,
        "oracle_service_beam_top_k": 4,
        "conflict_top_k2": 3,
        "conflict_sinr_threshold_db": 0.0,
    },
    "scheduler": {
        "domain_mode": "per_site_joint",
        "objective": "sum_rate",
        "max_mu_order": "auto",
        "cap_mu_order_by_rf": True,
        "algorithm": "greedy",
        "use_panel_constraint": True,
        "exhaustive_pruning": {
            "enabled": True,
            "sort_by_upper_bound": True,
            "zero_upper_bound": True,
            "branch_and_bound": True,
        },
        "conflict_penalty_lambda": 0.35,
        "unknown_interference_policy": "zero",
        "pf_tbar_init_mbps": 1.0,
    },
    "link_abstraction": {
        # v2 preferred path: Sionna SYS PHYAbstraction + ILLA. If sionna.sys or
        # torch is unavailable and sionna.fallback_to_numpy_if_unavailable=true,
        # a local precomputed-table fallback is used and recorded in status JSON.
        "mode": "sionna_sys_precomputed_bler",
        "mcs_table_index": 1,
        # Sionna SYS category 1 is downlink PDSCH. Category 0 is PUSCH.
        "mcs_category": 1,
        "sinr_mapping": "eesm",
        "eesm_beta_db": 5.0,
        "olla_enabled": True,
        "olla_step_db": 0.1,
        "harq_enabled": True,
        "bler_curve_slope": 1.1,
        "fallback_snr_min_db": -8.0,
        "fallback_snr_max_db": 35.0,
        "fallback_snr_step_db": 0.05,
    },
    "coverage_heatmap": {
        "enabled": True,
        "backend": "configured_channel_sampling",
        "grid_size": 40,
        "max_distance_m": 300.0,
        "chunk_size": 128,
        "fixed_vertical_beam_cdf": {
            "enabled": True,
            # "all" means every vertical DFT index of the active codebook.
            # For speed, this can also be a list, e.g. [0, 8, 16, 24].
            "candidate_v_indices": "all",
            "horizontal_num_beams": 4,
            "selection_metric": "mean_dbm"
        },
    },
    "progress": {
        "enabled": True,
    },
    "sionna": {
        "enable_import_probe": True,
        "prefer_sionna_sys_phy_abstraction": True,
        "fallback_to_numpy_if_unavailable": True,
        "device": None,
        "precision": None,
        "bs_polarization": "dual",
        "bs_polarization_type": "cross",
        "bs_antenna_pattern": "38.901",
        "ut_polarization": "single",
        "ut_polarization_type": "V",
        "ut_antenna_pattern": "omni",
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


def save_config(path: str | Path, cfg: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
