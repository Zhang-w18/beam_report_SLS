from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .sim import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-1 beam-domain SLS simulation.")
    parser.add_argument("--config", type=str, default="configs/phase1_single_cell.yaml",
                        help="YAML config path")
    parser.add_argument("--out", type=str, default="runs/phase1_demo",
                        help="Output directory")
    parser.add_argument("--num-drops", type=int, default=None,
                        help="Override system.num_drops for quick tests")
    parser.add_argument("--num-tti", type=int, default=None,
                        help="Override system.num_tti_per_drop")
    parser.add_argument("--olla-warmup-tti", type=int, default=None,
                        help="Override link_abstraction.olla_warmup_tti")
    parser.add_argument("--algorithm", type=str, default=None, choices=[
        "exhaustive",
        "greedy",
        "hard_conflict_greedy",
        "adaptive_lambda_greedy",
    ],
                        help="Override scheduler.algorithm")
    parser.add_argument("--domain-mode", type=str, default=None,
                        help="Override scheduler.domain_mode, e.g. per_site_joint or global")
    parser.add_argument("--layout", type=str, default=None,
                        help="Override topology.layout, e.g. three_site_triangle or seven_site_hex")
    parser.add_argument("--num-sites", type=int, default=None,
                        help="Override topology.num_sites")
    parser.add_argument("--objective", type=str, default=None, choices=["sum_rate", "proportional_fair"],
                        help="Override scheduler.objective")
    parser.add_argument("--skip-heatmap", action="store_true",
                        help="Disable coverage heatmap generation")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable progress output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_drops is not None:
        cfg["system"]["num_drops"] = args.num_drops
    if args.num_tti is not None:
        cfg["system"]["num_tti_per_drop"] = args.num_tti
    if args.olla_warmup_tti is not None:
        if args.olla_warmup_tti < 0:
            parser.error("--olla-warmup-tti must be >= 0")
        cfg["link_abstraction"]["olla_warmup_tti"] = args.olla_warmup_tti
    if args.algorithm is not None:
        matrix = cfg.get("evaluation", {}).get("matrix")
        if isinstance(matrix, dict):
            # A global CLI override applies to every enabled matrix row. The
            # capability validator will reject incompatible combinations.
            cfg["evaluation"]["matrix"] = {
                str(scheme): [args.algorithm] for scheme in matrix
            }
        else:
            cfg["scheduler"]["algorithm"] = args.algorithm
    if args.domain_mode is not None:
        cfg["scheduler"]["domain_mode"] = args.domain_mode
    if args.layout is not None:
        cfg["topology"]["layout"] = args.layout
    if args.num_sites is not None:
        cfg["topology"]["num_sites"] = args.num_sites
    if args.objective is not None:
        cfg["scheduler"]["objective"] = args.objective
    if args.skip_heatmap:
        cfg["coverage_heatmap"]["enabled"] = False
    if args.quiet:
        cfg.setdefault("progress", {})["enabled"] = False

    summary = run_simulation(cfg, Path(args.out))
    print("Simulation finished. Summary:")
    for scheme, vals in summary.items():
        if isinstance(vals, dict) and not str(scheme).startswith("_"):
            gain = vals.get("gain_over_baseline")
            gain_text = "n/a" if gain is None else f"{float(gain):+7.2%}"
            print(f"  {scheme:24s} avg_system={vals.get('avg_system_goodput_mbps', 0.0):9.3f} Mbps "
                  f"p05_ue={vals.get('p05_ue_goodput_mbps', 0.0):8.3f} Mbps "
                  f"oracle_ratio={vals.get('oracle_ratio', 0.0):6.3f} "
                  f"gain_base={gain_text:>8s}")
    print(f"Outputs written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
