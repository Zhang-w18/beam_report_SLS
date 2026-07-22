import numpy as np

from beam_sls.codebook import (ArrayConfig, build_network_tx_beams,
                               dft_codebook_from_array, distance_range_vertical_samples,
                               steering_vector_from_array)


def test_requested_trp_array_config():
    cfg = {
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
        "num_beams_h": 4,
        "num_beams_v": 4,
        "max_beams": 16,
    }
    a = ArrayConfig.from_dict(cfg)
    assert a.num_txru == 4
    assert a.num_ant == 1024
    assert a.expected_ae == 1024
    assert a.num_h == 16
    assert a.num_v == 32
    assert a.num_beams_h == 4
    assert a.num_beams_v == 4
    cb = dft_codebook_from_array(a, max_beams=16)
    assert cb.shape == (16, 1024)
    sv = steering_vector_from_array(a, 0.0, 0.0)
    assert sv.shape == (1024,)


def test_ue_array_can_use_same_3gpp_notation():
    cfg = {
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
        "num_beams_h": 4,
        "num_beams_v": 4,
        "max_beams": 16,
    }
    a = ArrayConfig.from_dict(cfg)
    assert a.num_txru == 4  # stored as generic RF-chain metadata
    assert a.num_ant == 16
    assert a.expected_ae == 16
    assert a.num_h == 4
    assert a.num_v == 4
    cb = dft_codebook_from_array(a, max_beams=16)
    assert cb.shape == (16, 16)


def test_panel_independent_codebook_and_fixed_vertical():
    from beam_sls.codebook import build_network_tx_beams
    cfg = {
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
        "beam_scope": "per_panel",
        "sampling_mode": "uniform",
        "num_beams_h": 4,
        "num_beams_v": 4,
        "max_beams": 16,
    }
    a = ArrayConfig.from_dict(cfg)
    assert a.normalized_beam_scope == "per_panel"
    assert a.num_array_panels == 2
    assert a.full_codebook_size == 512
    assert a.per_panel_codebook_size == 256
    ids, beams = build_network_tx_beams(num_cells=3, panels_per_cell=2, tx_cfg=a,
                                        max_beams_per_panel=16, site_id_by_cell=[0, 0, 0])
    assert beams.shape == (96, 1024)
    assert len(ids) == 3 * 2 * 16
    assert {b.array_panel_index for b in ids} == {0, 1}

    ids_fixed, beams_fixed = build_network_tx_beams(num_cells=3, panels_per_cell=2, tx_cfg=a,
                                                    max_beams_per_panel=4, site_id_by_cell=[0, 0, 0],
                                                    fixed_v_index=3)
    assert beams_fixed.shape == (3 * 2 * 4, 1024)
    assert {b.v_index for b in ids_fixed} == {3}


def test_rf_architecture_default_and_fully_connected():
    from beam_sls.config import load_config
    from beam_sls.rf import resolve_rf_architecture, resolved_max_mu_order

    cfg = load_config(None)
    tx = ArrayConfig.from_dict(cfg["tx_array"])
    rf = resolve_rf_architecture(cfg, tx)
    assert rf.connectivity == "panel_polarization_subarray"
    assert rf.tx_units_per_trp == 4
    assert rf.max_parallel_beams_per_trp == 4
    assert resolved_max_mu_order(cfg, rf) == 4
    assert {(u.array_panel_index, u.polarization_index) for u in rf.tx_units} == {(0, 0), (0, 1), (1, 0), (1, 1)}

    cfg["rf_architecture"]["txru_connectivity"] = "fully_connected"
    rf2 = resolve_rf_architecture(cfg, tx)
    assert rf2.connectivity == "fully_connected"
    assert rf2.effective_beam_scope == "joint"
    assert rf2.max_parallel_beams_per_trp == 4
    assert all(u.array_panel_index is None for u in rf2.tx_units)


def test_three_site_global_36ue_config_exposes_36_tx_units():
    import numpy as np

    from beam_sls.codebook import build_network_tx_beams
    from beam_sls.config import load_config
    from beam_sls.rf import resolve_rf_architecture, resolved_max_mu_order, tx_units_per_sector
    from beam_sls.topology import make_topology

    cfg = load_config("configs/v2_three_site_global_36ue.yaml")
    topo = make_topology(cfg, np.random.default_rng(1))
    tx = ArrayConfig.from_dict(cfg["tx_array"])
    rf = resolve_rf_architecture(cfg, tx)
    site_ids = [topo.sector_by_cell(c).site_id for c in range(topo.num_cells)]
    beam_ids, _ = build_network_tx_beams(
        num_cells=topo.num_cells,
        panels_per_cell=tx_units_per_sector(cfg, rf),
        tx_cfg=tx,
        max_beams_per_panel=int(cfg["tx_array"]["max_beams"]),
        site_id_by_cell=site_ids,
        rf_architecture=rf,
    )

    assert len(topo.sites) == 3
    assert topo.num_cells == 9
    assert len(topo.ues) == 36
    assert rf.tx_units_per_trp == 4
    assert resolved_max_mu_order(cfg, rf) == 36
    assert cfg["feedback"]["service_beam_top_k1"] == 56
    assert len({b.panel_key() for b in beam_ids}) == 36
    assert len(beam_ids) == 36 * 8


def test_distance_range_vertical_codebook_has_downward_mainlobes():
    cfg = {
        "num_h": 1,
        "num_v": 16,
        "dH": 0.5,
        "dV": 0.5,
        "num_beams_h": 1,
        "num_beams_v": 4,
        "vertical_beam_mode": "distance_range",
        "vertical_beam": {
            "min_horizontal_distance_m": 35.0,
            "max_horizontal_distance_m": 250.0,
            "height_difference_m": 23.5,
        },
    }
    array = ArrayConfig.from_dict(cfg)
    samples = distance_range_vertical_samples(35.0, 250.0, 23.5, 4, 0.5)
    codebook = dft_codebook_from_array(array, max_beams=4)

    phases = np.asarray([x["vertical_phase_rad"] for x in samples])
    assert codebook.shape == (4, 16)
    assert np.all(phases < 0.0)
    assert np.allclose(np.diff(phases), np.diff(phases)[0])
    assert np.isclose(samples[0]["horizontal_distance_m"], 35.0)
    assert np.isclose(samples[-1]["horizontal_distance_m"], 250.0)

    elevation_grid_deg = np.linspace(-60.0, 20.0, 8001)
    for beam, sample in zip(codebook, samples):
        responses = np.asarray([
            abs(np.vdot(steering_vector_from_array(array, 0.0, np.deg2rad(el)), beam))
            for el in elevation_grid_deg
        ])
        peak_elevation_deg = float(elevation_grid_deg[int(np.argmax(responses))])
        assert sample["elevation_deg"] < 0.0
        assert sample["downtilt_deg"] > 0.0
        assert peak_elevation_deg < 0.0
        assert abs(peak_elevation_deg - sample["elevation_deg"]) < 0.02


def test_distance_range_beam_metadata_is_written_to_beam_ids():
    array = ArrayConfig.from_dict({
        "num_h": 1,
        "num_v": 8,
        "num_beams_h": 1,
        "num_beams_v": 4,
        "vertical_beam_mode": "distance_range",
        "vertical_beam": {
            "min_horizontal_distance_m": 35.0,
            "max_horizontal_distance_m": 250.0,
            "bs_height_m": 25.0,
            "ue_height_m": 1.5,
        },
    })
    beam_ids, beams = build_network_tx_beams(1, 1, array, 4)

    assert beams.shape == (4, 8)
    assert all(beam.v_index is None for beam in beam_ids)
    assert all(beam.vertical_phase_rad < 0.0 for beam in beam_ids)
    assert all(beam.elevation_deg < 0.0 for beam in beam_ids)
    assert all(beam.downtilt_deg > 0.0 for beam in beam_ids)
    assert np.isclose(beam_ids[0].horizontal_distance_m, 35.0)
    assert np.isclose(beam_ids[-1].horizontal_distance_m, 250.0)


def test_distance_range_config_inherits_scenario_and_topology(tmp_path):
    from beam_sls.config import load_config

    config_path = tmp_path / "distance_range.yaml"
    config_path.write_text("tx_array:\n  vertical_beam_mode: distance_range\n", encoding="utf-8")
    cfg = load_config(config_path)
    array = ArrayConfig.from_dict(cfg["tx_array"])

    assert array.vertical_min_distance_m == cfg["scenario"]["min_ue_distance_m"]
    assert array.vertical_max_distance_m == cfg["scenario"]["max_ue_distance_m"]
    assert array.vertical_height_difference_m == cfg["topology"]["bs_height_m"] - cfg["topology"]["ue_height_m"]


def test_numpy_geometric_channel_uses_downward_bs_to_ue_elevation():
    from beam_sls.channel import generate_numpy_geometric_channel
    from beam_sls.topology import Sector, Site, Topology, UE

    cfg = {
        "scenario": {
            "carrier_frequency_ghz": 30.0,
            "num_clusters": 1,
            "delay_spread_ns": 1.0,
            "shadow_fading_std_db": 0.0,
            "pathloss_exponent": 2.0,
        },
        "measurement": {"num_freq_points": 1},
        "system": {"bandwidth_mhz": 20.0},
        "trp": {"num_trps_per_sector": 1},
        "rf_architecture": {"txru_connectivity": "fully_connected", "num_txru": 1},
    }
    tx_array = ArrayConfig(num_h=1, num_v=16, num_txru=1)
    rx_array = ArrayConfig(num_h=1, num_v=1)
    topology = Topology(
        ues=[UE(0, 100.0, 0.0, z_m=1.5, serving_cell=0, site_id=0)],
        sites=[Site(0, 0.0, 0.0, 25.0)],
        sectors=[Sector(0, 0, 0, 0.0)],
        carrier_frequency_ghz=30.0,
        isd_m=500.0,
    )
    channel = generate_numpy_geometric_channel(
        topology, cfg, tx_array, rx_array, np.random.default_rng(4)
    )

    elevation_grid_deg = np.linspace(-60.0, 20.0, 8001)
    responses = []
    for elevation_deg in elevation_grid_deg:
        beam = steering_vector_from_array(tx_array, 0.0, np.deg2rad(elevation_deg))
        responses.append(abs(channel.h_freq[0, 0, 0, 0] @ beam))
    peak_elevation_deg = float(elevation_grid_deg[int(np.argmax(responses))])

    assert peak_elevation_deg < 0.0


def test_distance_range_default_rf_path_builds_normalized_downward_beams():
    from beam_sls.rf import resolve_rf_architecture

    array = ArrayConfig.from_dict({
        "model": "tr38901_panel",
        "num_txru": 4,
        "M": 16, "N": 16, "P": 2, "Mg": 2, "Ng": 1, "Mp": 1, "Np": 1,
        "dH": 0.5, "dV": 0.5,
        "beam_scope": "per_panel",
        "num_beams_h": 4,
        "num_beams_v": 4,
        "vertical_beam_mode": "distance_range",
        "vertical_beam": {
            "min_horizontal_distance_m": 35.0,
            "max_horizontal_distance_m": 250.0,
            "height_difference_m": 23.5,
        },
    })
    cfg = {"rf_architecture": {
        "txru_connectivity": "panel_polarization_subarray",
        "allow_independent_polarization_beams": True,
        "num_txru": 4,
    }}
    rf = resolve_rf_architecture(cfg, array)
    beam_ids, beams = build_network_tx_beams(1, 4, array, 16, [0], rf_architecture=rf)

    assert beams.shape == (64, 1024)
    assert np.allclose(np.linalg.norm(beams, axis=1), 1.0)
    assert len({round(beam.vertical_phase_rad, 12) for beam in beam_ids}) == 4
    assert all(beam.vertical_phase_rad < 0.0 and beam.elevation_deg < 0.0 for beam in beam_ids)
