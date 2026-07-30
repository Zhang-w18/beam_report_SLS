import csv
from types import SimpleNamespace

import numpy as np

from beam_sls.config import load_config
from beam_sls.channel import DopplerChannelEvolver
from beam_sls.scheduler import update_pf_throughput
from beam_sls import sim


def _cfg(speed_kmh=3.0):
    return {
        "scenario": {"carrier_frequency_ghz": 30.0},
        "ue_drop": {"speed_kmh": speed_kmh},
        "pdsch": {"slot_duration_ms": 0.125},
        "scheduler": {
            "pf_averaging_window_tti": 100,
            "pf_tbar_init_mbps": 1.0,
        },
    }


def test_pf_ewma_updates_scheduled_and_unscheduled_ues():
    cfg = _cfg()
    cfg["scheduler"]["pf_averaging_window_tti"] = 10
    tbar = {0: 10.0, 1: 10.0}

    update_pf_throughput(tbar, [0, 1], {0: 30.0}, cfg)

    assert np.isclose(tbar[0], 12.0)
    assert np.isclose(tbar[1], 9.0)


def test_zero_speed_keeps_small_scale_channel_constant():
    cfg = _cfg(speed_kmh=0.0)
    h = np.ones((2, 1, 2, 2, 2), dtype=np.complex128)
    evolver = DopplerChannelEvolver.from_drop(
        h, cfg, np.random.default_rng(7)
    )

    np.testing.assert_array_equal(evolver.advance(), h)


def test_nonzero_speed_changes_only_the_channel_not_input_storage():
    cfg = _cfg(speed_kmh=30.0)
    h = np.ones((1, 1, 2, 2, 2), dtype=np.complex128)
    original = h.copy()
    evolver = DopplerChannelEvolver.from_drop(
        h, cfg, np.random.default_rng(7)
    )

    advanced = evolver.advance()

    np.testing.assert_array_equal(h, original)
    assert not np.array_equal(advanced, original)
    assert advanced.shape == original.shape


def test_continuous_tti_measures_once_and_schedules_each_tti(
        tmp_path, monkeypatch):
    cfg = load_config(None)
    cfg["scenario"]["channel_model"] = "numpy_geometric_uma"
    cfg["system"]["num_drops"] = 1
    cfg["system"]["continuous_tti"] = {"enabled": True, "duration_ms": 0.375}
    cfg["ue_drop"]["num_ut_per_sector"] = 1
    cfg["feedback"]["schemes"] = ["baseline"]
    cfg["analysis"]["baseline_no_interference_upper_bound"] = False
    cfg["measurement"]["num_freq_points"] = 2
    cfg["tx_array"].update({"num_beams_h": 1, "num_beams_v": 1, "max_beams": 1})
    cfg["ue_array"].update({"num_beams_h": 1, "num_beams_v": 1, "max_beams": 1})
    cfg["coverage_heatmap"]["enabled"] = False

    class Adapter:
        target_bler = 0.1
        num_mcs = 29
        status = SimpleNamespace(
            backend="unit_test", status="OK", target_bler=0.1,
            mcs_table_index=1, mcs_category=1,
        )

        @staticmethod
        def select_mcs_from_sinr_lin(_sinr):
            return 3

        @staticmethod
        def select_mcs_from_sinr_db(_sinr):
            return 3

        @staticmethod
        def is_outage_from_sinr_lin(_sinr, _mcs):
            return False

        @staticmethod
        def rate_mbps(mcs):
            return float(mcs * 10)

        @staticmethod
        def tbler_from_sinr_db(_sinr, _mcs):
            return 0.0

        @staticmethod
        def tbs_bits(mcs):
            return int(mcs * 1000)

    adapter = Adapter()
    monkeypatch.setattr(sim, "make_link_adapter", lambda _cfg: adapter)
    monkeypatch.setattr(
        sim, "make_scheduler_link_adapter", lambda backing, _cfg: backing
    )

    calls = {"measurement": 0, "reports": 0, "schedule": 0}
    original_measurement = sim.compute_gamma_measurement
    original_make_reports = sim.make_reports
    original_schedule = sim.schedule

    def counted_measurement(*args, **kwargs):
        calls["measurement"] += 1
        return original_measurement(*args, **kwargs)

    def counted_schedule(*args, **kwargs):
        calls["schedule"] += 1
        return original_schedule(*args, **kwargs)

    def counted_make_reports(*args, **kwargs):
        calls["reports"] += 1
        return original_make_reports(*args, **kwargs)

    monkeypatch.setattr(sim, "compute_gamma_measurement", counted_measurement)
    monkeypatch.setattr(sim, "make_reports", counted_make_reports)
    monkeypatch.setattr(sim, "schedule", counted_schedule)
    out_dir = tmp_path / "continuous"
    sim.run_simulation(cfg, out_dir)

    assert calls == {"measurement": 1, "reports": 1, "schedule": 3}
    with (out_dir / "metrics" / "reports.csv").open(
            newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (out_dir / "metrics" / "schedules.csv").open(
            newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_continuous_tti_count_and_warmup_count_override_duration():
    cfg = load_config(None)
    cfg["system"]["continuous_tti"].update({
        "enabled": True,
        "duration_ms": 2.0,
        "num_tti": 7,
        "warmup_tti": 3,
    })

    enabled, measured, warmup, duration_ms = sim.resolve_tti_counts(cfg)

    assert enabled
    assert measured == 7
    assert warmup == 3
    assert np.isclose(duration_ms, 7 * cfg["pdsch"]["slot_duration_ms"])
