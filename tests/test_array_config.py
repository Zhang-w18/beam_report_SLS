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
    assert rf.allow_independent_polarization_beams is False
    assert rf.dynamic_beam_assignment is True
    assert rf.tx_units_per_trp == 2
    assert rf.max_parallel_beams_per_trp == 2
    assert resolved_max_mu_order(cfg, rf) == 6
    assert len(rf.tx_units) == 1
    assert rf.tx_units[0].txru_index is None
    assert rf.tx_units[0].polarization_index is None
    assert rf.measurement_panel_index == 0
    assert rf.compact_panel_channel is True

    cfg["rf_architecture"]["txru_connectivity"] = "fully_connected"
    rf2 = resolve_rf_architecture(cfg, tx)
    assert rf2.connectivity == "fully_connected"
    assert rf2.effective_beam_scope == "joint"
    assert rf2.max_parallel_beams_per_trp == 4
    assert all(u.array_panel_index is None for u in rf2.tx_units)


def test_dynamic_trp_codebook_is_not_bound_to_txru():
    from beam_sls.codebook import build_network_tx_beams
    from beam_sls.config import load_config
    from beam_sls.rf import resolve_rf_architecture, tx_units_per_sector

    cfg = load_config(None)
    tx = ArrayConfig.from_dict(cfg["tx_array"])
    rf = resolve_rf_architecture(cfg, tx)
    beam_ids, beams = build_network_tx_beams(
        num_cells=3,
        panels_per_cell=tx_units_per_sector(cfg, rf),
        tx_cfg=tx,
        max_beams_per_panel=16,
        site_id_by_cell=[0, 0, 0],
        rf_architecture=rf,
    )

    assert beams.shape == (3 * 16, 512)
    assert len({beam.trp_key() for beam in beam_ids}) == 3
    assert all(beam.txru_index is None for beam in beam_ids)
    assert all(beam.polarization_index is None for beam in beam_ids)
    assert all(beam.beam_scope == "per_panel" for beam in beam_ids)
    assert all(beam.array_panel_index == 0 for beam in beam_ids)
    assert all(beam.codebook_size == 16 * 16 for beam in beam_ids)


def test_shared_codebook_can_measure_on_selected_reference_panel():
    from beam_sls.codebook import build_network_tx_beams
    from beam_sls.config import load_config
    from beam_sls.rf import resolve_rf_architecture

    cfg = load_config(None)
    cfg["measurement"]["tx_panel_index"] = 1
    tx = ArrayConfig.from_dict(cfg["tx_array"])
    rf = resolve_rf_architecture(cfg, tx)
    beam_ids, beams = build_network_tx_beams(
        1, rf.tx_units_per_trp, tx, 16, [0], rf_architecture=rf
    )

    assert rf.measurement_panel_index == 1
    assert all(beam.array_panel_index == 1 for beam in beam_ids)
    assert all(beam.txru_index is None for beam in beam_ids)
    assert beams.shape == (16, 512)
    # Compact ordering retains both polarizations of the 16x16 panel.
    assert np.all(np.count_nonzero(np.abs(beams) > 0.0, axis=1) == 512)


def test_full_channel_is_retained_and_panel_view_matches_full_calculation():
    import copy

    from beam_sls.channel import generate_channel
    from beam_sls.codebook import (
        build_network_tx_beams,
        dft_codebook_from_array,
        extract_panel_tx_dimension,
        panel_ae_indices,
    )
    from beam_sls.config import load_config
    from beam_sls.measurement import compute_gamma_measurement
    from beam_sls.rf import resolve_rf_architecture
    from beam_sls.topology import make_topology

    cfg = load_config(None)
    cfg["scenario"]["channel_model"] = "numpy_geometric_uma"
    cfg["ue_drop"]["num_ut_per_sector"] = 1
    cfg["measurement"]["num_freq_points"] = 2
    cfg["measurement"]["tx_panel_index"] = 1
    full_cfg = copy.deepcopy(cfg)
    full_cfg["measurement"]["use_panel_channel_views"] = False

    tx = ArrayConfig.from_dict(cfg["tx_array"])
    rx = ArrayConfig.from_dict(cfg["ue_array"])
    topo = make_topology(cfg, np.random.default_rng(3))
    retained = generate_channel(topo, cfg, tx, rx, np.random.default_rng(4))
    full = generate_channel(topo, full_cfg, tx, rx, np.random.default_rng(4))
    indices = panel_ae_indices(tx, 1)
    site_ids = [sec.site_id for sec in topo.sectors]
    compact_rf = resolve_rf_architecture(cfg, tx)
    full_rf = resolve_rf_architecture(full_cfg, tx)
    compact_ids, compact_beams = build_network_tx_beams(
        topo.num_cells, compact_rf.tx_units_per_trp, tx, 4, site_ids,
        rf_architecture=compact_rf,
    )
    full_ids, full_beams = build_network_tx_beams(
        topo.num_cells, full_rf.tx_units_per_trp, tx, 4, site_ids,
        rf_architecture=full_rf,
    )
    rx_beams = dft_codebook_from_array(rx, max_beams=2)

    assert retained.h_freq.shape[-1] == 1024
    assert full.h_freq.shape[-1] == 1024
    assert np.allclose(retained.h_freq, full.h_freq)
    panel_view = extract_panel_tx_dimension(retained.h_freq, tx, 1)
    assert panel_view.shape[-1] == 512
    assert np.shares_memory(panel_view, retained.h_freq) is False
    assert np.allclose(panel_view, full.h_freq[..., indices])
    assert np.allclose(compact_beams, full_beams[..., indices])

    compact_meas = compute_gamma_measurement(
        panel_view, compact_beams, rx_beams, compact_ids, 1.0, 0.1
    )
    full_meas = compute_gamma_measurement(
        full.h_freq, full_beams, rx_beams, full_ids, 1.0, 0.1
    )
    assert np.allclose(compact_meas.service_power_w, full_meas.service_power_w)
    assert np.allclose(compact_meas.gamma, full_meas.gamma)


def test_sionna_cir_axes_and_panelarray_antenna_order_are_explicitly_mapped():
    from beam_sls.channel import sionna_cir_to_internal_frequency_response
    from beam_sls.codebook import sionna_panelarray_source_indices

    tx = ArrayConfig.from_dict({
        "model": "tr38901_panel",
        "M": 2, "N": 2, "P": 2,
        "Mg": 2, "Ng": 1, "Mp": 1, "Np": 1,
    })
    rx = ArrayConfig.from_dict({
        "model": "tr38901_panel",
        "M": 1, "N": 2, "P": 1,
        "Mg": 1, "Ng": 1, "Mp": 1, "Np": 1,
    })
    a = np.zeros((1, 1, rx.num_ant, 1, tx.num_ant, 2, 2), dtype=np.complex128)
    for r in range(rx.num_ant):
        for n in range(tx.num_ant):
            base = 100.0 * r + n
            a[0, 0, r, 0, n, 0, 0] = base
            a[0, 0, r, 0, n, 1, 0] = 2.0 * base
            # Must be ignored because the static simulator explicitly uses T=0.
            a[0, 0, r, 0, n, :, 1] = 9999.0 + base
    tau = np.asarray([[[[0.0, 0.25]]]], dtype=float)
    freqs = np.asarray([-1.0, 0.0, 1.0])
    h = sionna_cir_to_internal_frequency_response(
        a, tau, freqs, tx, rx, time_index=0
    )
    rx_src = sionna_panelarray_source_indices(rx)
    tx_src = sionna_panelarray_source_indices(tx)
    base_expected = np.empty((rx.num_ant, tx.num_ant), dtype=np.complex128)
    for r_local, r_source in enumerate(rx_src):
        for n_local, n_source in enumerate(tx_src):
            base_expected[r_local, n_local] = 100.0 * r_source + n_source
    assert h.shape == (1, 1, 3, rx.num_ant, tx.num_ant)
    for fi, freq in enumerate(freqs):
        expected = base_expected * (
            1.0 + 2.0 * np.exp(-1j * 2.0 * np.pi * freq * 0.25)
        )
        assert np.allclose(h[0, 0, fi], expected)


def test_actual_transmission_assigns_scheduled_beams_to_distinct_panel_views():
    from beam_sls.codebook import BeamId, panel_ae_indices
    from beam_sls.link import realized_sinr_grid
    from beam_sls.measurement import MeasurementResult
    from beam_sls.scheduler import ScheduleResult, ScheduledLink

    tx = ArrayConfig.from_dict({
        "model": "tr38901_panel",
        "M": 1, "N": 2, "P": 2,
        "Mg": 2, "Ng": 1, "Mp": 1, "Np": 1,
    })
    schedule = ScheduleResult(
        scheme="test",
        objective_value=0.0,
        links=[
            ScheduledLink(0, 0, 0.0, 0, 0.0),
            ScheduledLink(1, 1, 0.0, 0, 0.0),
        ],
    )
    h = np.zeros((2, 1, 1, 1, tx.num_ant), dtype=np.complex128)
    h[:, 0, 0, 0, panel_ae_indices(tx, 0)] = 1.0
    h[:, 0, 0, 0, panel_ae_indices(tx, 1)] = 2.0
    beams = np.ones((2, 4), dtype=np.complex128) / 2.0
    beam_ids = [
        BeamId(cell=0, trp=0, panel=0, beam=i, global_index=i, tx_unit=0)
        for i in range(2)
    ]
    meas = MeasurementResult(
        service_power_w=np.ones((2, 2)),
        interference_power_w=np.zeros((2, 2, 2)),
        gamma=np.ones((2, 2, 2)),
        noise_power_w=1.0,
        selected_rx_beam=np.zeros((2, 2), dtype=int),
        su_mcs=np.zeros((2, 2), dtype=int),
        su_snr_db=np.zeros((2, 2)),
    )
    sinr = realized_sinr_grid(
        schedule,
        h,
        beams,
        np.ones((1, 1), dtype=np.complex128),
        beam_ids,
        meas,
        1.0,
        ignore_interference=True,
        tx_array=tx,
    )
    assert np.allclose(sinr[0], [4.0])
    assert np.allclose(sinr[1], [16.0])


def test_seven_site_three_trp_topology_and_capacity():
    import numpy as np

    from beam_sls.config import load_config
    from beam_sls.rf import resolve_rf_architecture, resolved_max_mu_order
    from beam_sls.topology import make_topology

    cfg = load_config("configs/v2_seven_site_hex.yaml")
    topo = make_topology(cfg, np.random.default_rng(1))
    tx = ArrayConfig.from_dict(cfg["tx_array"])
    rf = resolve_rf_architecture(cfg, tx)

    assert len(topo.sites) == 7
    assert topo.num_cells == 21
    assert all(sum(sec.site_id == site.site_id for sec in topo.sectors) == 3 for site in topo.sites)
    assert rf.max_parallel_beams_per_trp == tx.num_array_panels == 2
    # per_site_joint schedules three TRPs at a time.
    assert resolved_max_mu_order(cfg, rf) == 6


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
        "system": {"subcarrier_spacing_khz": 120.0},
        "pdsch": {"num_prbs": 132},
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
