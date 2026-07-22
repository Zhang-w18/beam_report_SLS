import copy

import numpy as np

from beam_sls.codebook import BeamId
from beam_sls.config import load_config
from beam_sls.feedback import ServiceCandidate, UEReport
from beam_sls.link_adaptation import BaseLinkAdapter, SchedulerLinkLookup
from beam_sls.scheduler import schedule


class _PiecewiseAdapter:
    target_bler = 0.1

    @staticmethod
    def select_mcs_from_sinr_lin(sinr_lin):
        return int(np.searchsorted([0.5, 1.0, 2.0, 4.0], float(sinr_lin), side="right"))

    @classmethod
    def select_mcs_from_sinr_db(cls, sinr_db):
        return cls.select_mcs_from_sinr_lin(10.0 ** (float(sinr_db) / 10.0))

    @staticmethod
    def is_outage_from_sinr_lin(sinr_lin, _mcs=None):
        return bool(float(sinr_lin) < 0.25)

    @classmethod
    def is_outage_from_sinr_db(cls, sinr_db, mcs_index=None):
        return cls.is_outage_from_sinr_lin(10.0 ** (float(sinr_db) / 10.0), mcs_index)

    @staticmethod
    def rate_mbps(mcs):
        return float((int(mcs) + 1) * 10.0)

    @classmethod
    def map_sinr_lin(cls, values):
        x = np.asarray(values, dtype=float)
        mcs = np.searchsorted([0.5, 1.0, 2.0, 4.0], x, side="right").astype(np.int32)
        outage = x < 0.25
        rates = np.where(outage, 0.0, (mcs + 1) * 10.0)
        return mcs, outage, rates


class _LookupReferenceAdapter(BaseLinkAdapter):
    def select_mcs_from_sinr_db(self, sinr_db, num_allocated_re=None):
        return 0 if sinr_db < 0.0 else (1 if sinr_db < 10.0 else 2)

    def tbler_from_sinr_db(self, sinr_db, mcs_index, num_allocated_re=None):
        return 1.0 if sinr_db < -5.0 else 0.0

    def rate_mbps(self, mcs_index):
        return float((int(mcs_index) + 1) * 7.0)


def _beam_ids(count):
    return [
        BeamId(cell=0, trp=0, panel=i, beam=0, global_index=i, tx_unit=i)
        for i in range(count)
    ]


def test_scheduler_lookup_matches_reference_across_decision_boundaries():
    cfg = load_config(None)
    cfg["scheduler"]["link_lookup"].update({
        "sinr_min_db": -20.0,
        "sinr_max_db": 20.0,
        "scan_step_db": 0.25,
        "refine_tolerance_db": 1e-8,
        "boundary_guard_db": 1e-5,
    })
    reference = _LookupReferenceAdapter(cfg)
    lookup = SchedulerLinkLookup(reference, cfg)
    points = np.asarray([-30.0, -5.000001, -5.0, -4.999999, -0.000001,
                         0.0, 0.000001, 9.999999, 10.0, 10.000001, 30.0])
    mcs, outage, rates = lookup.map_sinr_db(points)
    expected_mcs, expected_outage = reference.schedule_map_from_sinr_db(points)
    expected_rates = np.asarray([
        0.0 if out else reference.rate_mbps(selected)
        for selected, out in zip(expected_mcs, expected_outage)
    ])
    assert np.array_equal(mcs, expected_mcs)
    assert np.array_equal(outage, expected_outage)
    assert np.array_equal(rates, expected_rates)
    assert lookup.status["num_boundaries"] == 3


def test_incremental_limited_feedback_matches_v29_reference_greedy():
    cfg = load_config(None)
    cfg["scheduler"].update({
        "domain_mode": "global",
        "use_panel_constraint": True,
        "conflict_penalty_lambda": 6.0,
    })
    cfg["_resolved"] = {"max_mu_order": 3}
    beams = _beam_ids(6)
    reports = [
        UEReport(0, "topk_conflict_id", [
            ServiceCandidate(0, 8.0, 7, conflict_beams={2}),
            ServiceCandidate(1, 7.0, 6),
        ]),
        UEReport(1, "topk_conflict_id", [
            ServiceCandidate(2, 7.0, 6),
            ServiceCandidate(3, 6.0, 5, conflict_beams={0}),
        ]),
        UEReport(2, "topk_conflict_id", [
            ServiceCandidate(4, 9.0, 8, conflict_beams={2}),
            ServiceCandidate(5, 5.0, 4),
        ]),
    ]
    optimized_cfg = copy.deepcopy(cfg)
    optimized_cfg["scheduler"]["optimized_greedy"] = True
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg["scheduler"]["optimized_greedy"] = False

    optimized = schedule(reports, beams, optimized_cfg, link_adapter=_PiecewiseAdapter())
    reference = schedule(reports, beams, reference_cfg, link_adapter=_PiecewiseAdapter())

    assert [(x.ue_id, x.beam_index) for x in optimized.links] == [
        (x.ue_id, x.beam_index) for x in reference.links
    ]
    assert optimized.objective_value == reference.objective_value
    assert optimized.metadata["stats"]["implementation"] == "v2.10_incremental_limited_feedback"


def test_incremental_full_gamma_matches_v29_reference_greedy():
    cfg = load_config(None)
    cfg["scheduler"].update({"domain_mode": "global", "use_panel_constraint": True})
    cfg["_resolved"] = {"max_mu_order": 3}
    beams = _beam_ids(8)
    reports = []
    noise = 0.2
    for ue in range(4):
        first = 2 * ue
        candidates = [
            ServiceCandidate(first, 8.0 - ue, 3),
            ServiceCandidate(first + 1, 7.5 - ue, 3),
        ]
        signal = np.asarray([1.0 + 0.05 * ue + 0.02 * b for b in range(8)], dtype=float)
        gamma = np.empty((8, 8), dtype=float)
        for serving in range(8):
            for interferer in range(8):
                interference = 0.0 if serving == interferer else 0.04 * (1 + ((serving + 2 * interferer + ue) % 7))
                gamma[serving, interferer] = signal[serving] / (noise + interference)
        reports.append(UEReport(
            ue_id=ue,
            scheme="full_gamma",
            candidates=candidates,
            full_gamma=gamma,
            full_service_power_w=signal,
            full_noise_power_w=noise,
        ))

    optimized_cfg = copy.deepcopy(cfg)
    optimized_cfg["scheduler"]["optimized_greedy"] = True
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg["scheduler"]["optimized_greedy"] = False
    adapter = _PiecewiseAdapter()
    optimized = schedule(reports, beams, optimized_cfg, link_adapter=adapter)
    reference = schedule(reports, beams, reference_cfg, link_adapter=adapter)

    assert [(x.ue_id, x.beam_index) for x in optimized.links] == [
        (x.ue_id, x.beam_index) for x in reference.links
    ]
    assert optimized.objective_value == reference.objective_value
    assert optimized.metadata["stats"]["interference_matrix_elements"] == 64
    assert optimized.metadata["stats"]["implementation"] == "v2.10_incremental_vectorized_full_gamma"
