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
                        help="Override measured TTI count per drop in either TTI mode")
    parser.add_argument("--warmup-tti", "--olla-warmup-tti",
                        dest="warmup_tti", type=int, default=None,
                        help="Override warmup TTI count per drop")
    parser.add_argument("--algorithm", type=str, default=None, choices=[
        "exhaustive",
        "greedy",
        "hard_conflict_greedy",
        "adaptive_lambda_greedy",
    ],
                        help="Override scheduler.algorithm")
    parser.add_argument("--domain-mode", type=str, default=None,
                        help="Deprecated alias for scheduler.cluster_mode")
    parser.add_argument("--cluster-mode", type=str, default=None,
                        choices=["per_cell", "per_site", "global", "custom"],
                        help="Override scheduler.cluster_mode")
    parser.add_argument("--layout", type=str, default=None,
                        help="Override topology.layout, e.g. three_site_triangle or seven_site_hex")
    parser.add_argument("--num-sites", type=int, default=None,
                        help="Override topology.num_sites")
    parser.add_argument("--objective", type=str, default=None, choices=["sum_rate", "proportional_fair"],
                        help="Override scheduler.objective")
    parser.add_argument("--gamma-backend", type=str, default=None, choices=["numpy", "cupy", "auto"],
                        help="Override measurement.gamma_backend (CPU/GPU Gamma computation)")
    parser.add_argument("--skip-heatmap", action="store_true",
                        help="Disable coverage heatmap generation")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable progress output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_drops is not None:
        cfg["system"]["num_drops"] = args.num_drops
    if args.num_tti is not None:
        if args.num_tti <= 0:
            parser.error("--num-tti must be > 0")
        cfg["system"]["num_tti_per_drop"] = args.num_tti
        if bool(cfg["system"].get("continuous_tti", {}).get("enabled", False)):
            cfg["system"]["continuous_tti"]["num_tti"] = args.num_tti
    if args.warmup_tti is not None:
        if args.warmup_tti < 0:
            parser.error("--warmup-tti must be >= 0")
        cfg["link_abstraction"]["olla_warmup_tti"] = args.warmup_tti
        if bool(cfg["system"].get("continuous_tti", {}).get("enabled", False)):
            cfg["system"]["continuous_tti"]["warmup_tti"] = args.warmup_tti
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
        aliases = {
            "per_sector_independent": "per_cell",
            "per_site_joint": "per_site",
            "global": "global",
        }
        cfg["scheduler"]["cluster_mode"] = aliases.get(
            args.domain_mode, args.domain_mode
        )
    if args.cluster_mode is not None:
        cfg["scheduler"]["cluster_mode"] = args.cluster_mode
    if args.layout is not None:
        cfg["topology"]["layout"] = args.layout
    if args.num_sites is not None:
        cfg["topology"]["num_sites"] = args.num_sites
    if args.objective is not None:
        cfg["scheduler"]["objective"] = args.objective
    if args.gamma_backend is not None:
        cfg["measurement"]["gamma_backend"] = args.gamma_backend
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
