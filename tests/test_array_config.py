from beam_sls.codebook import ArrayConfig, dft_codebook_from_array, steering_vector_from_array


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
