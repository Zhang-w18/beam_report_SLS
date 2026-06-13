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
    parser.add_argument("--algorithm", type=str, default=None, choices=["exhaustive", "greedy"],
                        help="Override scheduler.algorithm")
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
    if args.algorithm is not None:
        cfg["scheduler"]["algorithm"] = args.algorithm
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
            print(f"  {scheme:24s} avg_system={vals.get('avg_system_goodput_mbps', 0.0):9.3f} Mbps "
                  f"p05_ue={vals.get('p05_ue_goodput_mbps', 0.0):8.3f} Mbps "
                  f"oracle_ratio={vals.get('oracle_ratio', 0.0):6.3f} "
                  f"gain_base={vals.get('gain_over_baseline', 0.0):+7.2%}")
    print(f"Outputs written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
