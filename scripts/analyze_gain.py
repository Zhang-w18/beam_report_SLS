#!/usr/bin/env python3
"""Post-processing gain-space analyzer for Sionna SLS beam-management runs.

This script does NOT re-run any simulation. It reads the metrics already written
by ``beam_sls.run`` (summary.json, schedules.csv, link_tti.csv, drops.csv,
resolved_config.yaml) and prints a text report that answers the questions in
docs/gain_space_analysis_report.md:

  Experiment A: interference-free ceiling and how much of it full_gamma captures.
  Experiment B: realized effective-SINR / MCS-saturation distribution.
  Experiment C: how similar each scheme's schedule is to the baseline schedule
                (a direct test of whether topk/threshold actually behave
                differently from baseline).
  Config block: the parameters needed to judge whether two runs are comparable.

Usage:
    python scripts/analyze_gain.py RUN_DIR [RUN_DIR ...] [--out FILE]

Example (single run):
    python scripts/analyze_gain.py runs/v2_4_one_site_three_sector \
        --out gain_analysis_one_site.txt

Example (compare two runs, e.g. 1-site vs 3-site global):
    python scripts/analyze_gain.py \
        runs/v2_4_one_site_site_domain \
        runs/v2_4_three_site_global \
        --out gain_analysis_compare.txt

The report is printed to stdout AND written to --out (default:
gain_analysis_report.txt). Copy the txt back for analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# MCS saturation point of the surrogate table in beam_sls/mcs.py: MCS 28 needs
# ~28 dB effective SINR and yields the maximum spectral efficiency (8.4375).
MCS_SAT_SINR_DB = 28.0
MAX_MCS_INDEX = 28


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _fnum(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if math.isinf(v) or math.isnan(v):
            return v
        return v
    except (TypeError, ValueError):
        return default


def _finite(vals: List[float], lo: float = -100.0, hi: float = 200.0) -> List[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if v <= lo or v >= hi:
            continue
        out.append(v)
    return out


def _pct(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _mean(vals: List[float]) -> float:
    return statistics.fmean(vals) if vals else float("nan")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


# --------------------------------------------------------------------------- #
# config loading
# --------------------------------------------------------------------------- #
CONFIG_KEYS = [
    ("scheduler", "domain_mode"),
    ("scheduler", "algorithm"),
    ("scheduler", "objective"),
    ("scheduler", "max_mu_order"),
    ("scheduler", "cap_mu_order_by_rf"),
    ("scheduler", "conflict_penalty_lambda"),
    ("feedback", "service_beam_top_k1"),
    ("feedback", "oracle_service_beam_top_k"),
    ("feedback", "conflict_top_k2"),
    ("feedback", "conflict_sinr_threshold_db"),
    ("ue_drop", "num_ut_per_sector"),
    ("topology", "layout"),
    ("topology", "num_sites"),
    ("topology", "sectors_per_site"),
    ("topology", "isd_m"),
    ("system", "num_drops"),
    ("system", "num_tti_per_drop"),
    ("system", "tx_power_dbm"),
    ("system", "target_bler"),
    ("scenario", "channel_model"),
    ("scenario", "carrier_frequency_ghz"),
    ("pdsch", "num_prbs"),
    ("tx_array", "num_beams_h"),
    ("tx_array", "num_beams_v"),
    ("tx_array", "vertical_beam_mode"),
    ("tx_array", "fixed_v_index"),
    ("tx_array", "num_txru"),
    ("rf_architecture", "txru_connectivity"),
    ("rf_architecture", "num_txru"),
    ("rf_architecture", "allow_independent_polarization_beams"),
]


def load_config(run: Path) -> Dict[str, Any]:
    """Return a flat {section.key: value} dict from resolved_config.yaml.

    Uses PyYAML if available (present in the Sionna env). Falls back to the JSON
    summaries for the few resolved values that matter most.
    """
    out: Dict[str, Any] = {}
    cfg_path = run / "resolved_config.yaml"
    cfg = None
    try:
        import yaml  # type: ignore

        if cfg_path.exists():
            with cfg_path.open() as fh:
                cfg = yaml.safe_load(fh)
    except Exception:
        cfg = None

    if isinstance(cfg, dict):
        for section, key in CONFIG_KEYS:
            sec = cfg.get(section, {})
            if isinstance(sec, dict) and key in sec:
                out[f"{section}.{key}"] = sec.get(key)
        feedback = cfg.get("feedback", {})
        if isinstance(feedback, dict) and "schemes" in feedback:
            out["feedback.schemes"] = feedback.get("schemes")
    else:
        out["_config_note"] = "resolved_config.yaml not parsed (PyYAML missing?)"

    # Resolved values from JSON summaries (authoritative, not 'auto').
    arr = _read_json(run / "array_config_summary.json")
    if arr:
        if "resolved_max_mu_order" in arr:
            out["_resolved.max_mu_order"] = arr.get("resolved_max_mu_order")
        if "tx_effective_tx_units_per_sector" in arr:
            out["_resolved.tx_units_per_sector"] = arr.get("tx_effective_tx_units_per_sector")
        if "tx_total_beams_per_sector" in arr:
            out["_resolved.tx_beams_per_sector"] = arr.get("tx_total_beams_per_sector")
    rf = _read_json(run / "rf_architecture_summary.json")
    if rf:
        if "max_parallel_beams_per_trp" in rf:
            out["_resolved.max_parallel_beams_per_trp"] = rf.get("max_parallel_beams_per_trp")
        if "txru_connectivity" in rf:
            out["_resolved.txru_connectivity"] = rf.get("txru_connectivity")
    return out


# --------------------------------------------------------------------------- #
# per-run data model
# --------------------------------------------------------------------------- #
class RunData:
    def __init__(self, run: Path):
        self.run = run
        self.name = run.name
        self.config = load_config(run)
        self.summary = _read_json(run / "metrics" / "summary.json") or {}
        self.schedules = _read_csv(run / "metrics" / "schedules.csv")
        self.link_tti = _read_csv(run / "metrics" / "link_tti.csv")
        self.drops = _read_csv(run / "metrics" / "drops.csv")
        self.schemes = [s for s in self.summary.keys() if not s.startswith("_")]
        # schedule links per (scheme, drop): list of (ue_id, beam_index, pred_rate, pred_sinr, pred_mcs)
        self._links_by: Dict[Tuple[str, int], List[Tuple[int, int, float, float, int]]] = {}
        for row in self.schedules:
            scheme = row.get("scheme", "")
            drop = int(_fnum(row.get("drop", 0), 0))
            try:
                j = json.loads(row.get("schedule_json", "{}"))
            except ValueError:
                j = {}
            links = []
            for lk in j.get("links", []):
                links.append((
                    int(lk.get("ue_id", -1)),
                    int(lk.get("beam_index", -1)),
                    _fnum(lk.get("predicted_rate_mbps", 0.0), 0.0),
                    _fnum(lk.get("predicted_sinr_db", float("nan"))),
                    int(lk.get("predicted_mcs", -1)),
                ))
            self._links_by[(scheme, drop)] = links

    def drops_list(self) -> List[int]:
        return sorted({d for (_s, d) in self._links_by})

    def links(self, scheme: str, drop: int):
        return self._links_by.get((scheme, drop), [])

    # ---- per-scheme realized metrics from summary.json ----
    def realized(self, scheme: str) -> Dict[str, float]:
        s = self.summary.get(scheme, {})
        if not isinstance(s, dict):
            return {}
        return {
            "sys_goodput": _fnum(s.get("avg_system_goodput_mbps", 0.0), 0.0),
            "ue_goodput": _fnum(s.get("avg_ue_goodput_mbps", 0.0), 0.0),
            "p05_ue": _fnum(s.get("p05_ue_goodput_mbps", 0.0), 0.0),
            "eff_sinr": _fnum(s.get("avg_eff_sinr_db", float("nan"))),
            "tbler": _fnum(s.get("avg_tbler", float("nan"))),
            "ack": _fnum(s.get("ack_rate", float("nan"))),
            "oracle_ratio": _fnum(s.get("oracle_ratio", float("nan"))),
            "gain_over_baseline": _fnum(s.get("gain_over_baseline", float("nan"))),
        }

    # ---- interference-free ceiling from baseline SU predicted rates ----
    def predicted_sum_per_drop(self, scheme: str) -> List[float]:
        out = []
        for d in self.drops_list():
            lk = self.links(scheme, d)
            if lk:
                out.append(sum(x[2] for x in lk))
        return out

    def avg_num_scheduled(self, scheme: str) -> float:
        counts = [len(self.links(scheme, d)) for d in self.drops_list()
                  if (scheme, d) in self._links_by]
        return _mean([float(c) for c in counts]) if counts else float("nan")

    # ---- link_tti effective SINR / MCS ----
    def link_sinr(self, scheme: str) -> List[float]:
        return _finite([_fnum(r.get("effective_sinr_db")) for r in self.link_tti
                        if r.get("scheme") == scheme])

    def link_actual_mcs(self, scheme: str) -> List[float]:
        return [_fnum(r.get("actual_mcs")) for r in self.link_tti
                if r.get("scheme") == scheme and r.get("actual_mcs") not in (None, "")]


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #
def _order_schemes(schemes: List[str]) -> List[str]:
    pref = ["full_gamma", "baseline", "topk_conflict_id", "threshold_conflict_set"]
    return [s for s in pref if s in schemes] + [s for s in schemes if s not in pref]


def section_config(rd: RunData, w) -> None:
    w("-" * 78)
    w(f"[CONFIG] {rd.name}")
    w("-" * 78)
    c = rd.config
    def g(k, default="?"):
        return c.get(k, default)
    w(f"  domain_mode                = {g('scheduler.domain_mode')}")
    w(f"  algorithm                  = {g('scheduler.algorithm')}")
    w(f"  objective                  = {g('scheduler.objective')}")
    w(f"  max_mu_order (cfg / resolved) = {g('scheduler.max_mu_order')} / "
      f"{g('_resolved.max_mu_order')}")
    w(f"  cap_mu_order_by_rf         = {g('scheduler.cap_mu_order_by_rf')}")
    w(f"  max_parallel_beams_per_trp = {g('_resolved.max_parallel_beams_per_trp')}")
    w(f"  conflict_penalty_lambda    = {g('scheduler.conflict_penalty_lambda')}"
      f"      <-- compare against per-UE rate (~700-1100 Mbps)")
    w(f"  service_beam_top_k1        = {g('feedback.service_beam_top_k1')}"
      f"      <-- candidates for baseline/topk/threshold")
    w(f"  oracle_service_beam_top_k  = {g('feedback.oracle_service_beam_top_k')}"
      f"      <-- candidates for full_gamma (asymmetry if != k1)")
    w(f"  conflict_top_k2            = {g('feedback.conflict_top_k2')}")
    w(f"  conflict_sinr_threshold_db = {g('feedback.conflict_sinr_threshold_db')}")
    w(f"  layout / num_sites / sectors = {g('topology.layout')} / "
      f"{g('topology.num_sites')} / {g('topology.sectors_per_site')}")
    w(f"  num_ut_per_sector          = {g('ue_drop.num_ut_per_sector')}")
    w(f"  num_beams_h x num_beams_v  = {g('tx_array.num_beams_h')} x {g('tx_array.num_beams_v')}"
      f"   (vertical_mode={g('tx_array.vertical_beam_mode')}, fixed_v={g('tx_array.fixed_v_index')})")
    w(f"  tx_units/sector, beams/sector = {g('_resolved.tx_units_per_sector')}, "
      f"{g('_resolved.tx_beams_per_sector')}")
    w(f"  channel_model / fc_ghz     = {g('scenario.channel_model')} / {g('scenario.carrier_frequency_ghz')}")
    w(f"  num_drops x num_tti        = {g('system.num_drops')} x {g('system.num_tti_per_drop')}")
    w(f"  target_bler / tx_power_dbm = {g('system.target_bler')} / {g('system.tx_power_dbm')}")
    # loading indicator
    try:
        nsite = float(g("topology.num_sites"))
        nsec = float(g("topology.sectors_per_site"))
        nue = float(g("ue_drop.num_ut_per_sector"))
        mu = float(g("_resolved.max_mu_order"))
        dom = str(g("scheduler.domain_mode"))
        if "site" in dom:
            scheduled_total = mu * nsite
        elif "sector" in dom or "cell" in dom:
            scheduled_total = mu * nsite * nsec
        else:  # global
            scheduled_total = mu
        panels_total = nsite * nsec * float(g("_resolved.tx_units_per_sector", "nan"))
        ues_total = nsite * nsec * nue
        w("  --- derived loading ---")
        w(f"  candidate UEs (network)    = {ues_total:.0f}")
        w(f"  panels/TX-units (network)  = {panels_total:.0f}")
        w(f"  scheduled UEs/slot (approx)= {scheduled_total:.0f}"
          f"   => load = {100.0*scheduled_total/max(panels_total,1):.0f}% of panels, "
          f"{100.0*scheduled_total/max(ues_total,1):.0f}% of UEs")
    except (TypeError, ValueError):
        pass
    w("")


def section_realized(rd: RunData, w) -> None:
    w("-" * 78)
    w(f"[REALIZED METRICS] {rd.name}   (from summary.json)")
    w("-" * 78)
    w(f"  {'scheme':24s} {'sys_gp':>9s} {'p05_ue':>8s} {'eff_SINR':>9s} "
      f"{'ack':>5s} {'#sched':>7s} {'gain%':>7s} {'oracle':>7s}")
    for s in _order_schemes(rd.schemes):
        r = rd.realized(s)
        if not r:
            continue
        w(f"  {s:24s} {r['sys_goodput']:9.1f} {r['p05_ue']:8.1f} "
          f"{r['eff_sinr']:9.2f} {r['ack']:5.2f} {rd.avg_num_scheduled(s):7.2f} "
          f"{100.0*r['gain_over_baseline']:7.2f} {r['oracle_ratio']:7.3f}")
    w("  (sys_gp = avg system goodput Mbps; gain% vs baseline; oracle = ratio to full_gamma)")
    w("")


def section_ceiling(rd: RunData, w) -> None:
    """Experiment A: interference-free ceiling."""
    w("-" * 78)
    w(f"[EXPERIMENT A] Interference-free ceiling & capture   {rd.name}")
    w("-" * 78)
    if "baseline" not in rd.schemes:
        w("  baseline scheme missing; cannot compute ceiling.")
        w("")
        return
    base_pred = rd.predicted_sum_per_drop("baseline")
    if not base_pred:
        w("  no baseline schedule links found in schedules.csv; cannot compute ceiling.")
        w("")
        return
    ceiling = _mean(base_pred)  # SU predicted sum = interference-free sum-rate (target-BLER)
    r_base = rd.realized("baseline")
    r_fg = rd.realized("full_gamma") if "full_gamma" in rd.schemes else {}
    realized_base = r_base.get("sys_goodput", float("nan"))
    ack_base = r_base.get("ack", float("nan"))
    # align ceiling to the same BLER overhead the realized numbers carry
    ceiling_aligned = ceiling * ack_base if not math.isnan(ack_base) else ceiling
    w(f"  interference-free ceiling (baseline SU predicted sum) = {ceiling:8.1f} Mbps")
    w(f"    - ack-aligned ceiling (x baseline ack={ack_base:.2f})   = {ceiling_aligned:8.1f} Mbps")
    w(f"  realized baseline (with interference)                 = {realized_base:8.1f} Mbps")
    if realized_base > 0:
        loss = 100.0 * (ceiling_aligned - realized_base) / realized_base
        w(f"  => throughput baseline LOSES to interference          = {loss:7.2f} %"
          f"   (this is the MAX gain any interference-aware scheme can add)")
        if r_fg:
            realized_fg = r_fg.get("sys_goodput", float("nan"))
            achieved = 100.0 * (realized_fg - realized_base) / realized_base
            w(f"  realized full_gamma                                   = {realized_fg:8.1f} Mbps")
            w(f"  => full_gamma ACHIEVED gain                           = {achieved:7.2f} %")
            if loss > 1e-6:
                w(f"  => CAPTURE RATIO (achieved / max)                     = {achieved/loss:7.2%}")
                w("     capture ~100%  -> full_gamma near the physical ceiling; small gain is REAL/expected.")
                w("     capture <<100% -> gain left on the table -> suspect greedy suboptimality (see Exp F/G).")
    w("  NOTE: ceiling uses baseline's max-SNR beams with interference removed; because")
    w("        baseline already picks max-SU-rate users, this is a valid upper bound on")
    w("        sum-rate. It is an estimate (target-BLER, no re-sim); see report for the")
    w("        exact re-simulation variant.")
    w("")


def section_saturation(rd: RunData, w) -> None:
    """Experiment B: effective SINR / MCS saturation distribution."""
    w("-" * 78)
    w(f"[EXPERIMENT B] Effective-SINR & MCS-saturation regime   {rd.name}")
    w("-" * 78)
    if not rd.link_tti:
        w("  link_tti.csv missing/empty; cannot compute SINR distribution.")
        w("")
        return
    w(f"  MCS saturates at ~{MCS_SAT_SINR_DB:.0f} dB (MCS {MAX_MCS_INDEX}, ~1122 Mbps). "
      f"Users above it cannot gain rate from interference avoidance.")
    w(f"  {'scheme':24s} {'p05':>6s} {'p50':>6s} {'p95':>6s} {'mean':>6s} "
      f"{'>=28dB':>7s} {'>=22dB':>7s} {'MCS=28':>7s}")
    for s in _order_schemes(rd.schemes):
        sinr = rd.link_sinr(s)
        mcs = rd.link_actual_mcs(s)
        if not sinr:
            continue
        n = len(sinr)
        f_sat = 100.0 * sum(1 for x in sinr if x >= MCS_SAT_SINR_DB) / n
        f_hi = 100.0 * sum(1 for x in sinr if x >= 22.0) / n
        f_mcsmax = (100.0 * sum(1 for m in mcs if m >= MAX_MCS_INDEX) / len(mcs)) if mcs else float("nan")
        w(f"  {s:24s} {_pct(sinr,5):6.1f} {_pct(sinr,50):6.1f} {_pct(sinr,95):6.1f} "
          f"{_mean(sinr):6.1f} {f_sat:6.1f}% {f_hi:6.1f}% {f_mcsmax:6.1f}%")
    w("  Interpretation: if a large fraction is >=28dB (MCS=28), the SUM-rate gain is")
    w("  compressed by saturation -> look at p05_ue / SINR-CDF instead of sum for the story.")
    w("")


def section_similarity(rd: RunData, w) -> None:
    """Experiment C: schedule similarity vs baseline."""
    w("-" * 78)
    w(f"[EXPERIMENT C] Schedule similarity vs baseline   {rd.name}")
    w("-" * 78)
    if "baseline" not in rd.schemes:
        w("  baseline scheme missing; cannot compare schedules.")
        w("")
        return
    drops = rd.drops_list()
    if not drops:
        w("  no schedules found.")
        w("")
        return

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        u = a | b
        return len(a & b) / len(u) if u else 1.0

    w("  Jaccard(scheme, baseline) over scheduled sets, averaged over drops.")
    w("  (ue,beam)=1.0 means IDENTICAL schedule to baseline -> the report info is unused.")
    w(f"  {'scheme':24s} {'J(ue,beam)':>11s} {'J(ue-set)':>10s} {'==baseline%':>11s}")
    for s in _order_schemes(rd.schemes):
        jub, ju, ident = [], [], []
        for d in drops:
            base = rd.links("baseline", d)
            cur = rd.links(s, d)
            if not base and not cur:
                continue
            base_ub = {(u, b) for (u, b, *_r) in base}
            cur_ub = {(u, b) for (u, b, *_r) in cur}
            base_u = {u for (u, b, *_r) in base}
            cur_u = {u for (u, b, *_r) in cur}
            jub.append(jaccard(cur_ub, base_ub))
            ju.append(jaccard(cur_u, base_u))
            ident.append(1.0 if cur_ub == base_ub else 0.0)
        if not jub:
            continue
        w(f"  {s:24s} {_mean(jub):11.3f} {_mean(ju):10.3f} {100.0*_mean(ident):10.1f}%")
    w("  Interpretation: topk/threshold with J~1.0 vs baseline => the conflict penalty is")
    w("  NOT changing decisions (see conflict_penalty_lambda). full_gamma with low J is")
    w("  where the gain comes from (it genuinely reschedules users/beams).")
    w("")


def section_cross_run(runs: List[RunData], w) -> None:
    if len(runs) < 2:
        return
    w("=" * 78)
    w("[CROSS-RUN COMPARISON]")
    w("=" * 78)
    w(f"  {'run':30s} {'domain':14s} {'muO':>4s} {'load%pan':>8s} "
      f"{'fg_gain%':>8s} {'capture%':>8s} {'fg_p05':>8s} {'base_p05':>8s}")
    for rd in runs:
        c = rd.config
        dom = str(c.get("scheduler.domain_mode", "?"))
        mu = c.get("_resolved.max_mu_order", "?")
        r_fg = rd.realized("full_gamma") if "full_gamma" in rd.schemes else {}
        r_base = rd.realized("baseline") if "baseline" in rd.schemes else {}
        gain = 100.0 * r_fg.get("gain_over_baseline", float("nan")) if r_fg else float("nan")
        # capture
        cap = float("nan")
        base_pred = rd.predicted_sum_per_drop("baseline")
        if base_pred and r_base:
            ceiling = _mean(base_pred) * r_base.get("ack", 1.0)
            rb = r_base.get("sys_goodput", float("nan"))
            rfg = r_fg.get("sys_goodput", float("nan")) if r_fg else float("nan")
            if rb > 0 and (ceiling - rb) > 1e-6:
                cap = 100.0 * (rfg - rb) / (ceiling - rb)
        # loading
        loadpan = float("nan")
        try:
            nsite = float(c.get("topology.num_sites"))
            nsec = float(c.get("topology.sectors_per_site"))
            muf = float(mu)
            if "site" in dom:
                sched = muf * nsite
            elif "sector" in dom or "cell" in dom:
                sched = muf * nsite * nsec
            else:
                sched = muf
            panels = nsite * nsec * float(c.get("_resolved.tx_units_per_sector"))
            loadpan = 100.0 * sched / max(panels, 1)
        except (TypeError, ValueError):
            pass
        w(f"  {rd.name[:30]:30s} {dom[:14]:14s} {str(mu):>4s} {loadpan:8.0f} "
          f"{gain:8.2f} {cap:8.1f} {r_fg.get('p05_ue', float('nan')):8.1f} "
          f"{r_base.get('p05_ue', float('nan')):8.1f}")
    w("  If two runs differ mainly in fg_gain% but have SAME load%pan and similar")
    w("  capture%, the gain difference is NOT a bug: it reflects greedy quality / regime.")
    w("  If load%pan differs a lot, the runs are NOT apples-to-apples -> fix before comparing.")
    w("")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Post-processing gain-space analyzer (no re-simulation).")
    ap.add_argument("run_dirs", nargs="+", help="one or more run output directories")
    ap.add_argument("--out", default="gain_analysis_report.txt", help="text report output path")
    args = ap.parse_args(argv)

    lines: List[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    w("=" * 78)
    w("Sionna SLS beam-management -- GAIN-SPACE ANALYSIS (post-processing, no re-sim)")
    w("=" * 78)
    w("")

    runs: List[RunData] = []
    for d in args.run_dirs:
        run = Path(d)
        if not run.exists():
            w(f"!! run dir not found: {run}")
            continue
        rd = RunData(run)
        runs.append(rd)
        section_config(rd, w)
        section_realized(rd, w)
        section_ceiling(rd, w)
        section_saturation(rd, w)
        section_similarity(rd, w)

    section_cross_run(runs, w)

    w("=" * 78)
    w("HOW TO READ / WHAT TO PASTE BACK:")
    w("  1. [CONFIG] blocks -> confirm the runs are comparable (same load%pan, k1 vs")
    w("     oracle_k, conflict_penalty_lambda).")
    w("  2. [EXPERIMENT A] capture ratio -> is 12/19% near the physical ceiling?")
    w("  3. [EXPERIMENT B] >=28dB fraction -> is sum-gain compressed by MCS saturation?")
    w("  4. [EXPERIMENT C] topk/threshold Jaccard vs baseline -> is the penalty inert?")
    w("  Paste this entire txt back for interpretation.")
    w("=" * 78)

    text = "\n".join(lines)
    print(text)
    try:
        Path(args.out).write_text(text + "\n")
        print(f"\n[written] {args.out}")
    except OSError as e:
        print(f"\n[warn] could not write {args.out}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
