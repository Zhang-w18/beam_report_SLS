import csv
from pathlib import Path

import numpy as np

from beam_sls.config import load_config
from beam_sls.sweep import _set_dotted, run_parameter_sweep
from beam_sls.utils import write_csv


def test_set_dotted_updates_existing_nested_key():
    cfg = {"system": {"tx_power_dbm": 33.0}}
    _set_dotted(cfg, "system.tx_power_dbm", 45.5)
    assert cfg["system"]["tx_power_dbm"] == 45.5


def test_parameter_sweep_reuses_seed_and_writes_combined_outputs(tmp_path, monkeypatch):
    seen = []

    def fake_run(config, run_dir):
        power = float(config["system"]["tx_power_dbm"])
        seen.append((power, config["system"]["random_seed"],
                     config["_sweep_progress_prefix"]))
        metrics = Path(run_dir) / "metrics"
        base_rows = [
            {"scheme": "baseline__greedy", "effective_sinr_db": power - 40,
             "actual_mcs": 10, "goodput_mbps": 100},
            {"scheme": "baseline__greedy", "effective_sinr_db": power - 39,
             "actual_mcs": 12, "goodput_mbps": 120},
        ]
        write_csv(metrics / "link_tti.csv", base_rows)
        write_csv(metrics / "ue_goodput.csv", [
            {"scheme": "baseline__greedy", "avg_goodput_mbps": 10},
        ])
        write_csv(metrics / "system_tti_goodput.csv", [
            {"scheme": "baseline__greedy", "system_goodput_mbps": 100},
        ])
        write_csv(metrics / "system_drop_avg_goodput.csv", [
            {"scheme": "baseline__greedy", "avg_system_goodput_mbps": 100},
        ])
        write_csv(metrics / "su_snr_samples.csv", [
            {"scheme": "baseline", "su_snr_db": 5},
        ])
        write_csv(metrics / "su_snr_max_per_ue.csv", [
            {"scheme": "baseline", "max_su_snr_db": 6},
        ])
        write_csv(metrics / "scheduled_ue_su_throughput.csv", [
            {"scheme": "baseline__greedy", "su_throughput_mbps": 90},
        ])
        return {"baseline__greedy": {
            "feedback_scheme": "baseline",
            "avg_system_goodput_mbps": power,
            "p05_system_goodput_mbps": power - 1,
            "avg_ue_goodput_mbps": 10.0,
            "p05_ue_goodput_mbps": 1.0,
            "avg_actual_mcs": 11.0,
            "p05_actual_mcs": 10.1,
            "p05_effective_sinr_db": 1.0,
            "p50_effective_sinr_db": 2.0,
            "p95_effective_sinr_db": 3.0,
        }}

    monkeypatch.setattr("beam_sls.sweep.run_simulation", fake_run)
    cfg = load_config(None)
    cfg["system"]["random_seed"] = 1234
    cfg["parameter_sweep"] = {
        "enabled": True,
        "parameter": "system.tx_power_dbm",
        "values": [43.0, 45.5],
    }
    cfg["feedback"]["schemes"] = ["baseline"]
    cfg["evaluation"] = {
        "matrix": {"baseline": ["greedy"]},
        "references": {"baseline": "baseline__greedy"},
    }
    rows = run_parameter_sweep(cfg, tmp_path)

    assert [(item[0], item[1]) for item in seen] == [(43.0, 1234), (45.5, 1234)]
    assert "[parameter 1/2" in seen[0][2]
    assert "[parameter 2/2" in seen[1][2]
    assert len(rows) == 2
    summary_path = tmp_path / "metrics" / "parameter_sweep_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert [float(row["parameter_value"]) for row in saved] == [43.0, 45.5]
    assert np.isclose(float(saved[1]["p05_actual_mcs"]), 10.1)
    assert (tmp_path / "figures" / "actual_mcs_cdf.png").exists()
    assert (tmp_path / "figures" / "ue_goodput_cdf.png").exists()
