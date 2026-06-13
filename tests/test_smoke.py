from pathlib import Path

import numpy as np

from beam_sls.codebook import BeamId
from beam_sls.config import load_config
from beam_sls.link import eesm
from beam_sls.link import run_tti_loop
from beam_sls.measurement import MeasurementResult
from beam_sls.scheduler import ScheduledLink, ScheduleResult
from beam_sls.sim import run_simulation


def test_smoke(tmp_path: Path):
    cfg = load_config(None)
    cfg["system"]["num_drops"] = 1
    cfg["system"]["num_tti_per_drop"] = 1
    cfg["ue_drop"]["num_ut_per_sector"] = 1
    cfg["measurement"]["num_freq_points"] = 4
    cfg["tx_array"]["num_beams_h"] = 2
    cfg["tx_array"]["num_beams_v"] = 2
    cfg["tx_array"]["max_beams"] = 4
    cfg["ue_array"]["num_beams_h"] = 2
    cfg["ue_array"]["num_beams_v"] = 2
    cfg["ue_array"]["max_beams"] = 4
    cfg["scheduler"]["algorithm"] = "greedy"
    cfg["coverage_heatmap"]["enabled"] = False
    summary = run_simulation(cfg, tmp_path / "out")
    assert "baseline" in summary
    assert (tmp_path / "out" / "metrics" / "summary.csv").exists()


def test_eesm_high_sinr_is_finite():
    val = eesm([1e12, 1e12, 1e12], beta_db=5.0)
    assert val == val
    assert val != float("inf")


def test_transmitted_mcs_uses_predicted_sinr_not_realized_sinr():
    cfg = load_config(None)
    cfg["system"]["num_tti_per_drop"] = 1
    cfg["link_abstraction"]["olla_enabled"] = True

    schedule = ScheduleResult(
        scheme="unit",
        objective_value=0.0,
        links=[ScheduledLink(
            ue_id=0,
            beam_index=0,
            predicted_sinr_db=-6.0,
            predicted_mcs=0,
            predicted_rate_mbps=0.0,
        )],
    )
    h_freq = np.ones((1, 1, 1, 1, 1), dtype=np.complex128) * 1e3
    tx_beams = np.ones((1, 1), dtype=np.complex128)
    rx_beams = np.ones((1, 1), dtype=np.complex128)
    beam_ids = [BeamId(cell=0, trp=0, panel=0, beam=0, global_index=0, tx_unit=0)]
    meas = MeasurementResult(
        service_power_w=np.ones((1, 1)),
        interference_power_w=np.zeros((1, 1, 1)),
        gamma=np.ones((1, 1, 1)),
        noise_power_w=1.0,
        selected_rx_beam=np.zeros((1, 1), dtype=int),
        su_mcs=np.zeros((1, 1), dtype=int),
        su_snr_db=np.zeros((1, 1)),
    )

    rows, _ = run_tti_loop(
        schedule, h_freq, tx_beams, rx_beams, beam_ids, meas,
        tx_power_w_per_panel=1.0, cfg=cfg, drop_idx=0,
        rng=np.random.default_rng(1),
    )

    assert rows[0].effective_sinr_db > 50.0
    assert rows[0].mcs_selection_sinr_db == -6.0
    assert rows[0].actual_mcs == 0
