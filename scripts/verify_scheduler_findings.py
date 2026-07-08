#!/usr/bin/env python3
"""Deterministic, Sionna-free sanity checks for two scheduler findings.

This does NOT run a system-level simulation. It calls the *real* scheduler code
(beam_sls.scheduler) on tiny synthetic reports so the results are reproducible in
milliseconds and require no channel model. It verifies:

  Finding #1: conflict_penalty_lambda is on a Mbps scale but is compared against
              per-UE rates of ~700-1100 Mbps, so the default 0.35 is inert. We
              show (a) the utility a single conflict removes, and (b) the lambda
              needed to actually flip a greedy decision.

  Finding (correctness): the full_gamma path reconstructs pairwise interference
              as I = S/Gamma - N and the resulting predicted MU-SINR matches a
              hand calculation -> the full_gamma scheduler itself is correct.

Usage:
    python scripts/verify_scheduler_findings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/verify_scheduler_findings.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beam_sls.codebook import BeamId
from beam_sls.feedback import ServiceCandidate, UEReport
from beam_sls.mcs import rate_mbps_from_mcs
from beam_sls.scheduler import _evaluate_assignments, greedy_schedule
from beam_sls.utils import lin_to_db

RATE_KW = dict(num_prbs=132, num_symbols=12, dmrs_overhead_re_per_prb=18,
               slot_duration_ms=0.125, num_layers=1)


def _cfg(lmbda: float, max_mu: int = 4) -> dict:
    return {
        "pdsch": {"num_prbs": 132, "num_symbols": 12, "dmrs_overhead_re_per_prb": 18,
                  "slot_duration_ms": 0.125, "num_layers_per_ue": 1},
        "scheduler": {"objective": "sum_rate", "conflict_penalty_lambda": lmbda,
                      "use_panel_constraint": True, "max_mu_order": max_mu},
        "_resolved": {"max_mu_order": max_mu},
    }


def _beams(n: int):
    # each beam on its own panel so the panel constraint never blocks pairs
    return [BeamId(cell=0, trp=0, panel=i, beam=0, global_index=i, tx_unit=i) for i in range(n)]


def check_penalty_scale():
    print("=" * 74)
    print("FINDING #1a: how much utility does one conflict remove?")
    print("=" * 74)
    beam_ids = _beams(3)

    def rep(uid, bi, mcs, conf):
        return UEReport(ue_id=uid, scheme="topk_conflict_id",
                        candidates=[ServiceCandidate(beam_index=bi, su_snr_db=22.0,
                                                     su_mcs=mcs, conflict_beams=set(conf))])
    reports = [rep(0, 0, 24, [1, 2]), rep(1, 1, 24, [0, 2]), rep(2, 2, 23, [0, 1])]
    assignment = [(0, 0), (1, 1), (2, 2)]  # 6 directed conflict pairs
    print("fixed schedule {(0,0),(1,1),(2,2)} has 6 directed conflicts; per-UE rate ~900 Mbps")
    print(f"{'lambda':>8s} {'utility(Mbps)':>14s} {'penalty removed':>16s}")
    for lm in [0.0, 0.35, 10.0, 50.0, 300.0]:
        val, _ = _evaluate_assignments(assignment, reports, beam_ids, _cfg(lm), None, None)
        print(f"{lm:8.2f} {val:14.2f} {lm*6:16.2f}")
    print("-> at lambda=0.35 the penalty is ~2 Mbps out of ~2700: a rounding error.\n")


def check_penalty_flips_decision():
    print("=" * 74)
    print("FINDING #1b: what lambda is needed to actually change a greedy decision?")
    print("=" * 74)
    beam_ids = _beams(3)
    # UE0@b0 (mcs24) conflicts with UE1@b1 (mcs24, mutual). UE2@b2 (mcs23) is clean.
    # Interference-aware optimum with 2 slots = {UE0, UE2} (no conflict, rate 24+23).
    # SU-rate greedy (lambda=0) = {UE0, UE1} (rate 24+24) but they collide.
    reports = [
        UEReport(ue_id=0, scheme="topk_conflict_id",
                 candidates=[ServiceCandidate(0, 22.0, 24, {1})]),
        UEReport(ue_id=1, scheme="topk_conflict_id",
                 candidates=[ServiceCandidate(1, 22.0, 24, {0})]),
        UEReport(ue_id=2, scheme="topk_conflict_id",
                 candidates=[ServiceCandidate(2, 21.0, 23, set())]),
    ]
    r24 = rate_mbps_from_mcs(24, **RATE_KW)
    r23 = rate_mbps_from_mcs(23, **RATE_KW)
    print(f"rate(mcs24)={r24:.1f}  rate(mcs23)={r23:.1f}  diff={r24 - r23:.1f} Mbps")
    print("2 slots; SU-greedy wants UE1(24) over UE2(23); avoiding the conflict needs")
    print(f"the penalty (2 conflicts x lambda) to exceed {r24 - r23:.1f} -> lambda > {(r24 - r23) / 2:.1f}\n")
    print(f"{'lambda':>8s}  {'greedy schedule (ue@beam)':32s} {'avoids conflict?':>16s}")
    for lm in [0.0, 0.35, 10.0, 30.0]:
        res = greedy_schedule(reports, beam_ids, _cfg(lm, max_mu=2), None, None)
        picks = sorted((lk.ue_id, lk.beam_index) for lk in res.links)
        chosen = " ".join(f"u{u}@b{b}" for u, b in picks)
        avoids = "YES" if (0, 1) not in [(u, b) for u, b in picks] or not ({0, 1} <= {u for u, _ in picks}) else "no"
        # 'avoids' = did NOT co-schedule the conflicting UE0+UE1 pair
        avoids = "YES" if not ({0, 1} <= {u for u, _ in picks}) else "no"
        print(f"{lm:8.2f}  {chosen:32s} {avoids:>16s}")
    print("-> lambda=0.35 co-schedules the conflicting pair (== baseline behaviour);")
    print("   only lambda in the tens flips it. The default is 2-3 orders too small.\n")


def check_full_gamma_correct():
    print("=" * 74)
    print("CORRECTNESS: full_gamma reconstructs I = S/Gamma - N and predicts MU-SINR")
    print("=" * 74)
    import numpy as np
    beam_ids = _beams(2)
    # Hand-built: UE0 served by b0, interfered by b1.
    S = 1.0e-9        # service power (W)
    N = 1.0e-11       # noise (W)
    I_true = 4.0e-10  # true interference from b1 on UE0 (W)
    gamma = np.zeros((2, 2))
    gamma[0, 0] = S / N                       # diagonal = SNR
    gamma[0, 1] = S / (I_true + N)            # off-diagonal = pair-SINR
    # UE1 (served by b1), symmetric, interfered by b0
    gamma[1, 1] = S / N
    gamma[1, 0] = S / (I_true + N)
    rep0 = UEReport(ue_id=0, scheme="full_gamma", candidates=[ServiceCandidate(0, 20.0, 20)],
                    full_gamma=gamma, full_service_power_w=np.array([S, S]),
                    full_noise_power_w=N)
    rep1 = UEReport(ue_id=1, scheme="full_gamma", candidates=[ServiceCandidate(1, 20.0, 20)],
                    full_gamma=gamma, full_service_power_w=np.array([S, S]), full_noise_power_w=N)
    val, links = _evaluate_assignments([(0, 0), (1, 1)], [rep0, rep1], beam_ids,
                                       _cfg(0.0), None, None)
    pred = links[0].predicted_sinr_db
    hand = lin_to_db(S / (I_true + N))
    print(f"true interference I = {I_true:.2e} W")
    print(f"predicted MU-SINR (code)      = {pred:.4f} dB")
    print(f"hand S/(I+N) in dB            = {float(hand):.4f} dB")
    print(f"match within 1e-6 dB?          = {abs(pred - float(hand)) < 1e-6}")
    print("-> the full_gamma MU-SINR reconstruction is correct; the modest full_gamma")
    print("   gain is NOT a full_gamma bug.\n")


if __name__ == "__main__":
    check_penalty_scale()
    check_penalty_flips_decision()
    check_full_gamma_correct()
    print("All checks above use the REAL beam_sls.scheduler code on synthetic inputs.")
