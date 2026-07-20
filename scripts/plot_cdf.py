#!/usr/bin/env python3
"""Draw configurable CDF figures from an existing simulation run.

This is an offline post-processing tool: it reads metrics CSV files and never
re-runs topology, channel, feedback, scheduling, or link simulation.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINE_NO_INTERFERENCE = "baseline_no_interference_upper_bound"

METRICS: Dict[str, Dict[str, str]] = {
    "link_goodput": {
        "file": "link_tti.csv",
        "column": "goodput_mbps",
        "xlabel": "TTI link goodput [Mbps]",
        "title": "Link goodput CDF",
    },
    "ue_goodput": {
        "file": "ue_goodput.csv",
        "column": "avg_goodput_mbps",
        "xlabel": "Per-UE average goodput [Mbps]",
        "title": "Per-UE average goodput CDF",
    },
    "effective_sinr": {
        "file": "link_tti.csv",
        "column": "effective_sinr_db",
        "xlabel": "Effective SINR [dB]",
        "title": "Effective SINR CDF",
    },
    "tbler": {
        "file": "link_tti.csv",
        "column": "tbler",
        "xlabel": "Transport-block error probability",
        "title": "TBLER CDF",
    },
    "olla_offset": {
        "file": "link_tti.csv",
        "column": "olla_offset_db",
        "xlabel": "OLLA backoff [dB]",
        "title": "OLLA offset CDF",
    },
    "reported_su_snr": {
        "file": "su_snr_samples.csv",
        "column": "su_snr_db",
        "xlabel": "Reported standalone SNR [dB]",
        "title": "Reported SU SNR CDF (all candidates)",
    },
    "reported_max_su_snr": {
        "file": "su_snr_max_per_ue.csv",
        "column": "max_su_snr_db",
        "xlabel": "Maximum reported standalone SNR per UE [dB]",
        "title": "Per-UE maximum reported SU SNR CDF",
    },
}

STYLE_KEYS = {
    "color",
    "linestyle",
    "linewidth",
    "marker",
    "markersize",
    "markevery",
    "alpha",
    "drawstyle",
    "zorder",
}

DEFAULT_LABELS = {
    "full_gamma": "Full Gamma (oracle)",
    "baseline": "Baseline",
    "topk_conflict_id": "Top-K conflict ID",
    "threshold_conflict_set": "Threshold conflict set",
    BASELINE_NO_INTERFERENCE: "Baseline SU (no interference)",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required metric file not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_pair(value: Any, name: str) -> Tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list")
    return float(value[0]), float(value[1])


def _load_plot_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required when --config is used") from exc
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Plot config must contain a YAML mapping")
    return data


def _curve_specs(config: Dict[str, Any], cli_schemes: Sequence[str] | None,
                 available: Sequence[str]) -> List[Dict[str, Any]]:
    if cli_schemes:
        existing = {
            str(item.get("case_id", item.get("scheme"))): dict(item)
            for item in config.get("curves", [])
            if isinstance(item, dict) and item.get("case_id", item.get("scheme")) is not None
        }
        specs = [existing.get(str(s), {"case_id": str(s)}) for s in cli_schemes]
    elif config.get("curves"):
        specs = [dict(item) for item in config["curves"] if isinstance(item, dict)]
    else:
        specs = [{"case_id": scheme} for scheme in available]

    requested = [str(item.get("case_id", item.get("scheme", ""))) for item in specs]
    missing = [scheme for scheme in requested if scheme not in available]
    if missing:
        raise ValueError(
            f"Schemes not present in metric CSV: {', '.join(missing)}. "
            f"Available: {', '.join(available)}"
        )
    return specs


def _extract_values(rows: Iterable[Dict[str, str]], column: str,
                    scheme: str, group_field: str = "scheme") -> np.ndarray:
    vals = []
    for row in rows:
        if row.get(group_field) != scheme:
            continue
        value = _finite_float(row.get(column))
        if value is not None:
            vals.append(value)
    return np.sort(np.asarray(vals, dtype=float))


def _write_ecdf_data(path: Path,
                     data: Sequence[Tuple[str, str, np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scheme", "label", "x", "cdf"])
        writer.writeheader()
        for scheme, label, x, y in data:
            for xx, yy in zip(x, y):
                writer.writerow({
                    "scheme": scheme,
                    "label": label,
                    "x": f"{float(xx):.12g}",
                    "cdf": f"{float(yy):.12g}",
                })


def draw_cdf(run_dir: Path, config: Dict[str, Any], args) -> Tuple[Path, Path | None]:
    metric = str(args.metric or config.get("metric", "ue_goodput"))
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {', '.join(METRICS)}")
    metric_cfg = METRICS[metric]
    rows = _read_csv(run_dir / "metrics" / metric_cfg["file"])
    report_metric = metric in {"reported_su_snr", "reported_max_su_snr"}
    preferred_group = "feedback_scheme" if report_metric else "case_id"
    group_field = preferred_group if rows and preferred_group in rows[0] else "scheme"
    available = list(dict.fromkeys(row.get(group_field, "") for row in rows if row.get(group_field)))

    if args.list_schemes or getattr(args, "list_cases", False):
        print("\n".join(available))
        raise SystemExit(0)

    requested_cases = getattr(args, "cases", None) or args.schemes
    specs = _curve_specs(config, requested_cases, available)
    font_size = float(args.font_size or config.get("font_size", 12.0))
    figsize = _as_pair(config.get("figsize", [7.5, 5.0]), "figsize")
    dpi = int(config.get("dpi", 180))
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": float(config.get("title_font_size", font_size + 1.0)),
        "axes.labelsize": float(config.get("label_font_size", font_size)),
        "xtick.labelsize": float(config.get("tick_font_size", font_size - 1.0)),
        "ytick.labelsize": float(config.get("tick_font_size", font_size - 1.0)),
        "legend.fontsize": float(config.get("legend_font_size", font_size - 1.0)),
    })

    fig, ax = plt.subplots(figsize=figsize)
    exported: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    for spec in specs:
        scheme = str(spec.get("case_id", spec.get("scheme")))
        x = _extract_values(rows, metric_cfg["column"], scheme, group_field=group_field)
        if x.size == 0:
            raise ValueError(f"No finite {metric_cfg['column']} samples for scheme {scheme}")
        y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
        label = str(spec.get("label", DEFAULT_LABELS.get(scheme, scheme)))
        style = {key: spec[key] for key in STYLE_KEYS if key in spec and spec[key] is not None}
        style.setdefault("linewidth", 2.0)
        style.setdefault("drawstyle", "steps-post")
        ax.plot(x, y, label=label, **style)
        exported.append((scheme, label, x, y))

    ax.set_xlabel(str(args.xlabel or config.get("xlabel", metric_cfg["xlabel"])))
    ax.set_ylabel(str(config.get("ylabel", "CDF")))
    ax.set_title(str(args.title or config.get("title", metric_cfg["title"])))
    xlim = _as_pair(config.get("xlim"), "xlim")
    ylim = _as_pair(config.get("ylim", [0.0, 1.0]), "ylim")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if bool(config.get("grid", True)):
        ax.grid(True, which=str(config.get("grid_which", "major")),
                alpha=float(config.get("grid_alpha", 0.3)))
    if bool(config.get("legend", True)):
        ax.legend(loc=str(config.get("legend_loc", "best")),
                  ncol=int(config.get("legend_ncol", 1)),
                  frameon=bool(config.get("legend_frame", True)))
    fig.tight_layout()

    output = Path(args.output or config.get("output") or
                  run_dir / "figures" / f"custom_{metric}_cdf.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    data_value = args.export_data or config.get("export_data")
    data_path = Path(data_value) if data_value else None
    if data_path is not None:
        _write_ecdf_data(data_path, exported)
    return output, data_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw configurable CDFs from an existing beam_sls run directory."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing metrics/*.csv")
    parser.add_argument("--config", help="Optional YAML plot configuration")
    parser.add_argument("--metric", choices=sorted(METRICS), help="CDF metric")
    parser.add_argument("--schemes", nargs="+", help="Legacy scheme/series IDs to draw")
    parser.add_argument("--cases", nargs="+", help="Evaluation case IDs to draw, in legend order")
    parser.add_argument("--output", help="Output PNG/PDF/SVG path")
    parser.add_argument("--export-data", help="Optional CSV path for the plotted ECDF points")
    parser.add_argument("--font-size", type=float, help="Base font size")
    parser.add_argument("--title", help="Figure title")
    parser.add_argument("--xlabel", help="X-axis label")
    parser.add_argument("--list-schemes", action="store_true",
                        help="List legacy schemes/series available for the metric and exit")
    parser.add_argument("--list-cases", action="store_true",
                        help="List evaluation cases available for the selected metric and exit")
    args = parser.parse_args()

    try:
        config = _load_plot_config(args.config)
        output, data_path = draw_cdf(args.run_dir, config, args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Figure written to: {output.resolve()}")
    if data_path is not None:
        print(f"ECDF data written to: {data_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
