from pathlib import Path

import numpy as np

from beam_sls.codebook import BeamId
from beam_sls.config import load_config
from beam_sls.feedback import ServiceCandidate, UEReport, make_reports
from beam_sls.link import eesm, realized_sinr_grid, run_tti_loop
from beam_sls.measurement import MeasurementResult, SparseGamma, compute_gamma_measurement
from beam_sls.scheduler import ScheduledLink, ScheduleResult, exhaustive_schedule, normalize_domain_mode, schedule
from beam_sls.sim import build_ue_goodput_rows, run_simulation, schedule_similarity_rows, summarize_su_snr
from beam_sls.topology import make_topology


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
    assert "baseline_no_interference_upper_bound" in summary
    assert (tmp_path / "out" / "metrics" / "ue_goodput.csv").exists()
    assert (tmp_path / "out" / "metrics" / "schedule_similarity.csv").exists()
    assert (tmp_path / "out" / "metrics" / "su_snr_samples.csv").exists()
    assert (tmp_path / "out" / "figures" / "ue_goodput_cdf.png").exists()


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


def test_no_interference_reference_forces_denominator_to_noise_only():
    schedule = ScheduleResult(
        scheme="baseline",
        objective_value=0.0,
        links=[
            ScheduledLink(0, 0, 0.0, 0, 0.0),
            ScheduledLink(1, 1, 0.0, 0, 0.0),
        ],
    )
    h_freq = np.ones((2, 2, 1, 1, 1), dtype=np.complex128)
    tx_beams = np.ones((2, 1), dtype=np.complex128)
    rx_beams = np.ones((1, 1), dtype=np.complex128)
    beam_ids = [
        BeamId(cell=0, trp=0, panel=i, beam=0, global_index=i, tx_unit=i)
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

    with_interference = realized_sinr_grid(
        schedule, h_freq, tx_beams, rx_beams, beam_ids, meas, 1.0,
    )
    noise_only = realized_sinr_grid(
        schedule, h_freq, tx_beams, rx_beams, beam_ids, meas, 1.0,
        ignore_interference=True,
    )

    assert np.allclose(with_interference[0], [0.5])
    assert np.allclose(noise_only[0], [1.0])


def test_multi_site_topology_layouts():
    cfg = load_config(None)
    cfg["ue_drop"]["num_ut_per_sector"] = 1

    cfg["topology"]["layout"] = "three_site_triangle"
    cfg["topology"]["num_sites"] = 3
    topo = make_topology(cfg, np.random.default_rng(1))
    assert len(topo.sites) == 3
    assert topo.num_cells == 9
    assert len(topo.ues) == 9
    d01 = np.hypot(topo.sites[0].x_m - topo.sites[1].x_m, topo.sites[0].y_m - topo.sites[1].y_m)
    d12 = np.hypot(topo.sites[1].x_m - topo.sites[2].x_m, topo.sites[1].y_m - topo.sites[2].y_m)
    d02 = np.hypot(topo.sites[0].x_m - topo.sites[2].x_m, topo.sites[0].y_m - topo.sites[2].y_m)
    assert np.allclose([d01, d12, d02], [500.0, 500.0, 500.0])

    cfg["topology"]["layout"] = "seven_site_hex"
    cfg["topology"]["num_sites"] = 7
    topo = make_topology(cfg, np.random.default_rng(2))
    assert len(topo.sites) == 7
    assert topo.num_cells == 21
    assert len(topo.ues) == 21


def test_site_domain_feedback_and_schedule():
    cfg = load_config(None)
    cfg["scheduler"]["algorithm"] = "greedy"
    cfg["scheduler"]["domain_mode"] = "per_site_joint"
    cfg["_resolved"] = {"max_mu_order": 1}
    beam_ids = [
        BeamId(cell=0, trp=0, panel=0, beam=0, global_index=0, tx_unit=0),
        BeamId(cell=1, trp=0, panel=0, beam=0, global_index=1, tx_unit=1),
        BeamId(cell=2, trp=1, panel=0, beam=0, global_index=2, tx_unit=2),
        BeamId(cell=3, trp=1, panel=0, beam=0, global_index=3, tx_unit=3),
    ]
    meas = MeasurementResult(
        service_power_w=np.ones((2, 4)),
        interference_power_w=np.zeros((2, 4, 4)),
        gamma=np.ones((2, 4, 4)),
        noise_power_w=1.0,
        selected_rx_beam=np.zeros((2, 4), dtype=int),
        su_mcs=np.asarray([[1, 2, 28, 27], [28, 27, 1, 2]], dtype=int),
        su_snr_db=np.asarray([[1.0, 2.0, 30.0, 29.0], [30.0, 29.0, 1.0, 2.0]], dtype=float),
    )
    reports = make_reports(
        meas, beam_ids, schemes=["baseline"], k1=1, oracle_top_k=1, k2=1, threshold_db=0.0,
        ue_site_ids={0: 0, 1: 1},
        ue_serving_cells={0: 0, 1: 2},
        candidate_beam_indices_by_ue={0: [0, 1], 1: [2, 3]},
    )["baseline"]

    assert reports[0].candidates[0].beam_index == 1
    assert reports[1].candidates[0].beam_index == 3
    sched = schedule(reports, beam_ids, cfg)
    assert sched.metadata["domain_mode"] == "per_site_joint"
    assert len(sched.links) == 2
    assert {beam_ids[l.beam_index].trp for l in sched.links} == {0, 1}


def test_single_site_three_sector_independent_is_sector_domain():
    cfg = load_config(None)
    cfg["scheduler"]["algorithm"] = "greedy"
    cfg["scheduler"]["domain_mode"] = "single_site_three_sector_independent"
    cfg["_resolved"] = {"max_mu_order": 1}
    assert normalize_domain_mode(cfg["scheduler"]["domain_mode"]) == "per_sector_independent"

    beam_ids = [
        BeamId(cell=0, trp=0, panel=0, beam=0, global_index=0, tx_unit=0),
        BeamId(cell=1, trp=0, panel=0, beam=0, global_index=1, tx_unit=1),
    ]
    meas = MeasurementResult(
        service_power_w=np.ones((2, 2)),
        interference_power_w=np.zeros((2, 2, 2)),
        gamma=np.ones((2, 2, 2)),
        noise_power_w=1.0,
        selected_rx_beam=np.zeros((2, 2), dtype=int),
        su_mcs=np.asarray([[1, 28], [28, 1]], dtype=int),
        su_snr_db=np.asarray([[1.0, 30.0], [30.0, 1.0]], dtype=float),
    )
    reports = make_reports(
        meas, beam_ids, schemes=["baseline"], k1=1, oracle_top_k=1, k2=1, threshold_db=0.0,
        ue_site_ids={0: 0, 1: 0},
        ue_serving_cells={0: 0, 1: 1},
        candidate_beam_indices_by_ue={0: [0], 1: [1]},
    )["baseline"]

    assert reports[0].candidates[0].beam_index == 0
    assert reports[1].candidates[0].beam_index == 1
    sched = schedule(reports, beam_ids, cfg)
    assert sched.metadata["domain_mode"] == "per_sector_independent"
    assert len(sched.links) == 2
    assert {l.beam_index for l in sched.links} == {0, 1}


def test_domain_limited_measurement_uses_sparse_gamma():
    beam_ids = [
        BeamId(cell=0, trp=0, panel=0, beam=0, global_index=0, tx_unit=0),
        BeamId(cell=1, trp=0, panel=0, beam=0, global_index=1, tx_unit=1),
    ]
    h_freq = np.ones((1, 2, 1, 1, 1), dtype=np.complex128)
    tx_beams = np.ones((2, 1), dtype=np.complex128)
    rx_beams = np.ones((1, 1), dtype=np.complex128)
    meas = compute_gamma_measurement(
        h_freq, tx_beams, rx_beams, beam_ids,
        tx_power_w_per_panel=1.0,
        noise_power_w=1.0,
        candidate_beam_indices_by_ue={0: [0]},
    )

    assert isinstance(meas.gamma, SparseGamma)
    assert meas.service_power_w[0, 0] > 0.0
    assert meas.service_power_w[0, 1] == 0.0
    assert meas.gamma[0, 0, 0] > 0.0
    assert meas.gamma[0, 0, 1] == 0.0


def test_exhaustive_pruning_matches_unpruned_small_case():
    cfg = load_config(None)
    cfg["scheduler"]["algorithm"] = "exhaustive"
    cfg["scheduler"]["domain_mode"] = "global"
    cfg["scheduler"]["use_panel_constraint"] = True
    cfg["_resolved"] = {"max_mu_order": 2}
    beam_ids = [
        BeamId(cell=0, trp=0, panel=0, beam=0, global_index=0, tx_unit=0),
        BeamId(cell=0, trp=0, panel=1, beam=0, global_index=1, tx_unit=1),
        BeamId(cell=1, trp=1, panel=0, beam=0, global_index=2, tx_unit=2),
        BeamId(cell=1, trp=1, panel=1, beam=0, global_index=3, tx_unit=3),
    ]
    reports = [
        UEReport(0, "baseline", [ServiceCandidate(0, 10.0, 10), ServiceCandidate(1, 9.0, 9)]),
        UEReport(1, "baseline", [ServiceCandidate(0, 8.0, 8), ServiceCandidate(2, 7.0, 7)]),
        UEReport(2, "baseline", [ServiceCandidate(2, 12.0, 12), ServiceCandidate(3, 6.0, 6)]),
        UEReport(3, "baseline", [ServiceCandidate(1, 11.0, 11), ServiceCandidate(3, 5.0, 5)]),
    ]
    cfg_pruned = load_config(None)
    cfg_pruned["scheduler"].update(cfg["scheduler"])
    cfg_pruned["_resolved"] = {"max_mu_order": 2}
    cfg_pruned["scheduler"]["exhaustive_pruning"] = {
        "enabled": True,
        "sort_by_upper_bound": True,
        "zero_upper_bound": True,
        "branch_and_bound": True,
    }
    cfg_unpruned = load_config(None)
    cfg_unpruned["scheduler"].update(cfg["scheduler"])
    cfg_unpruned["_resolved"] = {"max_mu_order": 2}
    cfg_unpruned["scheduler"]["exhaustive_pruning"] = {
        "enabled": False,
        "sort_by_upper_bound": False,
        "zero_upper_bound": False,
        "branch_and_bound": False,
    }

    pruned = exhaustive_schedule(reports, beam_ids, cfg_pruned)
    unpruned = exhaustive_schedule(reports, beam_ids, cfg_unpruned)
    assert np.isclose(pruned.objective_value, unpruned.objective_value)
    assert pruned.metadata["stats"]["evaluated_assignment_count"] <= unpruned.metadata["stats"]["evaluated_assignment_count"]


def test_hard_conflict_greedy_removes_candidate_not_entire_ue():
    cfg = load_config(None)
    cfg["scheduler"]["algorithm"] = "hard_conflict_greedy"
    cfg["scheduler"]["domain_mode"] = "global"
    cfg["scheduler"]["use_panel_constraint"] = True
    cfg["_resolved"] = {"max_mu_order": 2}
    beam_ids = [
        BeamId(cell=0, trp=0, panel=0, beam=0, global_index=0, tx_unit=0),
        BeamId(cell=0, trp=0, panel=1, beam=0, global_index=1, tx_unit=1),
        BeamId(cell=0, trp=0, panel=2, beam=0, global_index=2, tx_unit=2),
    ]
    reports = [
        UEReport(0, "topk_conflict_id", [ServiceCandidate(0, 20.0, 20)]),
        # The conflict is intentionally reported only in the reverse direction.
        UEReport(1, "topk_conflict_id", [
            ServiceCandidate(1, 15.0, 15, conflict_beams={0}),
            ServiceCandidate(2, 10.0, 10),
        ]),
    ]

    result = schedule(reports, beam_ids, cfg)

    assert {(l.ue_id, l.beam_index) for l in result.links} == {(0, 0), (1, 2)}
    assert result.metadata["stats"]["removed_conflicting_candidates"] == 1


def test_adaptive_lambda_uses_candidate_rate_median():
    class FakeAdapter:
        @staticmethod
        def rate_mbps(mcs):
            return float(mcs * 10.0)

    cfg = load_config(None)
    cfg["scheduler"]["algorithm"] = "adaptive_lambda_greedy"
    cfg["scheduler"]["domain_mode"] = "global"
    cfg["scheduler"]["adaptive_lambda_alpha"] = 0.2
    cfg["_resolved"] = {"max_mu_order": 1}
    beam_ids = [
        BeamId(cell=0, trp=0, panel=i, beam=0, global_index=i, tx_unit=i)
        for i in range(3)
    ]
    reports = [
        UEReport(0, "topk_conflict_id", [ServiceCandidate(0, 1.0, 1), ServiceCandidate(1, 2.0, 2)]),
        UEReport(1, "topk_conflict_id", [ServiceCandidate(2, 3.0, 3)]),
    ]

    result = schedule(reports, beam_ids, cfg, link_adapter=FakeAdapter())
    stats = result.metadata["stats"]

    assert stats["conflict_penalty_mode"] == "adaptive"
    assert stats["candidate_su_rate_median_mbps"] == 20.0
    assert stats["conflict_penalty_lambda_mbps"] == 4.0


def test_analysis_helpers_include_zero_ues_and_exact_pair_similarity():
    ue_goodput = build_ue_goodput_rows(
        [{"scheme": "a", "drop": 0, "tti": 0, "ue_id": 0, "goodput_mbps": 100.0}],
        ["a"],
        [{"drop": 0, "ue_id": 0}, {"drop": 0, "ue_id": 1}],
        num_tti=2,
    )
    assert [r["avg_goodput_mbps"] for r in ue_goodput] == [50.0, 0.0]

    by_drop, aggregate = schedule_similarity_rows(
        {(0, "a"): {(0, 1), (1, 2)}, (0, "b"): {(0, 1), (2, 3)}},
        ["a", "b"],
    )
    assert by_drop[0]["num_same_pairs"] == 1
    assert by_drop[0]["jaccard_similarity"] == 1.0 / 3.0
    assert aggregate[0]["micro_jaccard_similarity"] == 1.0 / 3.0

    maxima, snr_summary = summarize_su_snr([
        {"drop": 0, "scheme": "a", "ue_id": 0, "su_snr_db": 1.0},
        {"drop": 0, "scheme": "a", "ue_id": 0, "su_snr_db": 3.0},
        {"drop": 0, "scheme": "a", "ue_id": 1, "su_snr_db": 2.0},
    ], ["a"])
    assert len(maxima) == 2
    assert snr_summary[0]["num_reported_candidate_samples"] == 3
    assert snr_summary[0]["avg_reported_su_snr_db"] == 2.0
