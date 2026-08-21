import csv
import inspect

import numpy as np

from beam_sls.config import load_config
from beam_sls.ftp_traffic import (
    FTPArrivalOnlyConfig,
    count_arriving_ues_by_beam,
    count_ues_by_beam,
    sample_ue_arrivals,
)
from beam_sls.service_beam_statistics import (
    _cell_arrival_summary_rows,
    _global_collision_pmf_rows,
    _pmf_rows,
    _summary_rows,
    run_service_beam_statistics,
    select_best_service_beam,
)
from beam_sls.codebook import BeamId


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_arrivals_are_per_ue_and_a_ue_counts_once_per_window():
    traffic = FTPArrivalOnlyConfig(0.2, 0.5, 0.1)
    first = sample_ue_arrivals(np.random.default_rng(7), traffic, 4)
    second = sample_ue_arrivals(np.random.default_rng(7), traffic, 4)
    assert np.array_equal(first, second)
    counts = count_arriving_ues_by_beam(
        [10, 11, 12], [3, 0, 2], {10: 1, 11: 0, 12: 1}, 2
    )
    assert counts.tolist() == [0, 2]
    assert count_ues_by_beam([10, 11, 12], {10: 1, 11: 0, 12: 1}, 2).tolist() == [1, 2]


def test_best_beam_tie_uses_smallest_global_index():
    assert select_best_service_beam(np.array([3.0, 4.0, 4.0]), [2, 1]) == 1


def test_zero_arrival_rate_and_pmf_normalization():
    traffic = FTPArrivalOnlyConfig(0.0, 0.5, 0.1)
    samples = sample_ue_arrivals(np.random.default_rng(8), traffic, 100)
    assert np.count_nonzero(samples) == 0
    beams = [
        BeamId(cell=0, trp=0, panel=0, beam=0, global_index=0),
        BeamId(cell=0, trp=0, panel=0, beam=1, global_index=1),
    ]
    by_beam = {0: [0, 1, 0, 2], 1: [0, 0, 0, 0]}
    pmf = _pmf_rows(beams, by_beam, FTPArrivalOnlyConfig(1.0, 0.5, 0.1), 4)
    for beam_index in (0, 1):
        rows = [row for row in pmf if row["beam_index"] == beam_index]
        assert rows[0]["ue_count"] == 0
        assert sum(row["probability"] for row in rows) == 1.0
    zero_pmf = _pmf_rows(
        beams,
        {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]},
        traffic,
        4,
    )
    assert all(row["probability"] == 1.0 for row in zero_pmf if row["ue_count"] == 0)


def test_global_collision_pmf_pools_only_nonzero_beam_samples():
    rows, summary = _global_collision_pmf_rows({0: [0, 1, 2], 1: [1, 1, 0]})
    assert rows == [
        {"ue_count": 1, "sample_count": 3, "probability": 0.75},
        {"ue_count": 2, "sample_count": 1, "probability": 0.25},
    ]
    assert summary["num_beam_observation_samples"] == 6
    assert summary["num_nonzero_beam_observation_samples"] == 4
    assert summary["no_collision_probability"] == 0.75
    assert summary["collision_probability"] == 0.25


def test_cell_arrival_summary_reports_active_and_raw_arrivals():
    rows = _cell_arrival_summary_rows(
        {0: [1, 0, 2], 1: [0, 1, 0]},
        {0: [2, 0, 3], 1: [0, 4, 0]},
        3,
    )
    assert rows[0]["mean_arriving_ue_count"] == 1.0
    assert rows[0]["mean_num_arrivals"] == 5.0 / 3.0
    assert rows[1]["mean_active_ue_count"] == 1.0 / 3.0


def test_statistics_mode_does_not_contain_anonymous_multinomial_call():
    source = inspect.getsource(run_service_beam_statistics)
    assert "allocate_arrivals_to_beams" not in source


def test_statistics_smoke_binds_arrivals_to_cached_ues_and_writes_zero_rows(tmp_path):
    cfg = load_config("configs/v2_19_service_beam_statistics_smoke.yaml")
    summary = run_service_beam_statistics(cfg, tmp_path)
    metrics = tmp_path / "metrics"
    cache = _read_csv(metrics / "service_beam_ue_cache.csv")
    arrivals = _read_csv(metrics / "ftp_ue_arrival_samples.csv")
    candidate_counts = _read_csv(metrics / "service_beam_candidate_ue_count_samples.csv")
    candidate_pmf = _read_csv(metrics / "service_beam_candidate_ue_count_pmf.csv")
    active_counts = _read_csv(metrics / "service_beam_active_ue_count_samples.csv")
    active_pmf = _read_csv(metrics / "service_beam_active_ue_count_pmf.csv")
    global_pmf = _read_csv(metrics / "service_beam_global_collision_pmf.csv")
    global_summary = _read_csv(metrics / "service_beam_global_collision_summary.csv")
    cell_samples = _read_csv(metrics / "cell_arrival_ue_count_samples.csv")
    cell_summary = _read_csv(metrics / "cell_arrival_ue_count_summary.csv")
    assert summary["ftp_model"] == "3gpp_ftp_model_3_arrival_only"
    assert len(cache) == 210
    cache_by_key = {(int(row["drop"]), int(row["ue_id"])): row for row in cache}
    assert arrivals
    for row in arrivals:
        key = (int(row["drop"]), int(row["ue_id"]))
        assert key in cache_by_key
        assert row["best_service_beam_index"] == cache_by_key[key]["best_service_beam_index"]

    # Static collision distribution: all 210 candidate UEs are counted once.
    assert len(candidate_counts) == summary["num_beams"]
    assert sum(int(row["candidate_ue_count"]) for row in candidate_counts) == 210
    assert sum(int(row["candidate_ue_count"]) for row in candidate_counts) == len(cache)

    # 1 drop x 20 observations x all beams; zero active counts are explicit rows.
    assert len(active_counts) == 20 * summary["num_beams"]
    assert {int(row["active_ue_count"]) for row in active_counts} >= {0}
    for observation in range(20):
        observed = [row for row in arrivals if int(row["observation"]) == observation]
        counted = [row for row in active_counts if int(row["observation"]) == observation]
        assert sum(int(row["active_ue_count"]) for row in counted) == sum(
            int(row["has_arrival"]) for row in observed
        )
    for beam_index in {int(row["beam_index"]) for row in active_pmf}:
        rows = [row for row in active_pmf if int(row["beam_index"]) == beam_index]
        assert abs(sum(float(row["probability"]) for row in rows) - 1.0) < 1e-12
    for beam_index in {int(row["beam_index"]) for row in candidate_pmf}:
        rows = [row for row in candidate_pmf if int(row["beam_index"]) == beam_index]
        assert abs(sum(float(row["probability"]) for row in rows) - 1.0) < 1e-12
    assert global_pmf[0]["ue_count"] == "1"
    assert abs(sum(float(row["probability"]) for row in global_pmf) - 1.0) < 1e-12
    assert abs(
        float(global_summary[0]["no_collision_probability"])
        + float(global_summary[0]["collision_probability"])
        - 1.0
    ) < 1e-12
    assert abs(float(global_summary[0]["no_collision_probability"]) - sum(
        float(row["probability"]) for row in global_pmf if int(row["ue_count"]) == 1
    )) < 1e-12
    assert len(cell_samples) == 20 * 3
    assert len(cell_summary) == 3
    for observation in range(20):
        cell_rows = [
            row for row in cell_samples if int(row["observation"]) == observation
        ]
        assert sum(int(row["active_ue_count"]) for row in cell_rows) == sum(
            int(row["has_arrival"]) for row in arrivals
            if int(row["observation"]) == observation
        )


def test_default_mode_remains_scheduling():
    cfg = load_config(None)
    assert cfg["system"]["run_mode"] == "scheduling"
    assert cfg["service_beam_statistics"]["traffic"]["model"] == (
        "3gpp_ftp_model_3_arrival_only"
    )
