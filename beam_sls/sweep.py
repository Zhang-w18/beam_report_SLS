from __future__ import annotations

import copy
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .evaluation import resolve_evaluation_plan
from .plotting import plot_cdf
from .sim import run_simulation
from .utils import ensure_dir, write_csv, write_json


def _set_dotted(config: Dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part.strip() for part in dotted_path.split(".") if part.strip()]
    if not parts:
        raise ValueError("parameter_sweep.parameter must be a non-empty dotted path")
    node: Dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"parameter_sweep.parameter does not resolve to a config key: {dotted_path}")
        node = child
    if parts[-1] not in node:
        raise ValueError(f"parameter_sweep.parameter does not resolve to a config key: {dotted_path}")
    node[parts[-1]] = copy.deepcopy(value)


def _value_component(value: Any) -> str:
    text = str(value).strip().replace("-", "minus").replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return text or "value"


def _read_metric(path: Path, column: str, scheme: str) -> List[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            float(row[column])
            for row in csv.DictReader(handle)
            if row.get("scheme") == scheme and row.get(column, "") != ""
        ]


def _combined_plots(out_dir: Path, parameter: str,
                    runs: Sequence[Dict[str, Any]]) -> None:
    definitions = (
        ("effective_sinr_cdf.png", "link_tti.csv", "effective_sinr_db",
         "Effective SINR [dB]", "Effective SINR CDF", "case_id"),
        ("actual_mcs_cdf.png", "link_tti.csv", "actual_mcs",
         "Actual scheduled MCS index", "Actual scheduled MCS CDF", "case_id"),
        ("link_goodput_cdf.png", "link_tti.csv", "goodput_mbps",
         "TTI link goodput [Mbps]", "Link goodput CDF", "case_id"),
        ("ue_goodput_cdf.png", "ue_goodput.csv", "avg_goodput_mbps",
         "Per-UE average goodput [Mbps]", "Per-UE goodput CDF", "case_id"),
        ("system_tti_goodput_cdf.png", "system_tti_goodput.csv",
         "system_goodput_mbps", "System goodput per TTI [Mbps]",
         "Instantaneous system goodput CDF", "case_id"),
        ("system_drop_avg_goodput_cdf.png", "system_drop_avg_goodput.csv",
         "avg_system_goodput_mbps", "Per-drop average system goodput [Mbps]",
         "Per-drop measured-window average system goodput CDF", "case_id"),
        ("reported_su_snr_cdf.png", "su_snr_samples.csv", "su_snr_db",
         "Reported standalone SNR [dB]", "Reported SU SNR CDF",
         "feedback_scheme"),
        ("reported_max_su_snr_per_ue_cdf.png", "su_snr_max_per_ue.csv",
         "max_su_snr_db", "Maximum reported standalone SNR per UE [dB]",
         "Per-UE maximum reported SU SNR CDF", "feedback_scheme"),
        ("scheduled_ue_su_throughput_cdf.png",
         "scheduled_ue_su_throughput.csv", "su_throughput_mbps",
         "Scheduled-UE SU throughput [Mbps]",
         "Scheduled-UE standalone throughput CDF", "case_id"),
    )
    for filename, csv_name, column, xlabel, title, scheme_key in definitions:
        curves: Dict[str, List[float]] = {}
        for run in runs:
            label = f"{parameter}={run['parameter_value']}"
            curves[label] = _read_metric(
                Path(run["run_dir"]) / "metrics" / csv_name, column,
                str(run[scheme_key]),
            )
        plot_cdf(curves, xlabel, title, out_dir / "figures" / filename)


def run_parameter_sweep(cfg: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    sweep = cfg.get("parameter_sweep", {}) or {}
    parameter = str(sweep.get("parameter", "")).strip()
    values = list(sweep.get("values", []) or [])
    if not values:
        raise ValueError("parameter_sweep.values must contain at least one value")
    if str(cfg.get("system", {}).get("run_mode", "scheduling")).lower() != "scheduling":
        raise ValueError("parameter_sweep currently supports system.run_mode=scheduling only")
    case_ids = [case.case_id for case in resolve_evaluation_plan(cfg).cases]
    if len(case_ids) != 1:
        raise ValueError(
            "v2.20 parameter sweep requires exactly one evaluation case; "
            f"found {case_ids}"
        )

    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    base_seed = int(cfg.get("system", {}).get("random_seed", 1))
    leaf = parameter.rsplit(".", 1)[-1]
    for index, value in enumerate(values, start=1):
        run_cfg = copy.deepcopy(cfg)
        run_cfg["parameter_sweep"]["enabled"] = False
        _set_dotted(run_cfg, parameter, value)
        prefix = f"[parameter {index}/{len(values)} {parameter}={value}]"
        run_cfg["_sweep_progress_prefix"] = prefix
        run_dir = out_dir / f"{leaf}_{_value_component(value)}"
        print(f"{prefix} starting; common random_seed={base_seed}", flush=True)
        summary = run_simulation(run_cfg, run_dir)
        case_id = case_ids[0]
        metrics = summary[case_id]
        row = {
            "parameter_index": index,
            "parameter_count": len(values),
            "parameter": parameter,
            "parameter_value": value,
            "random_seed": base_seed,
            "case_id": case_id,
            "feedback_scheme": metrics.get("feedback_scheme", case_id),
            "avg_system_goodput_mbps": metrics["avg_system_goodput_mbps"],
            "p05_system_goodput_mbps": metrics["p05_system_goodput_mbps"],
            "avg_ue_goodput_mbps": metrics["avg_ue_goodput_mbps"],
            "p05_ue_goodput_mbps": metrics["p05_ue_goodput_mbps"],
            "avg_actual_mcs": metrics["avg_actual_mcs"],
            "p05_actual_mcs": metrics["p05_actual_mcs"],
            "p05_effective_sinr_db": metrics["p05_effective_sinr_db"],
            "p50_effective_sinr_db": metrics["p50_effective_sinr_db"],
            "p95_effective_sinr_db": metrics["p95_effective_sinr_db"],
            "run_dir": str(run_dir.resolve()),
        }
        rows.append(row)
        run_records.append({**row, "run_dir": run_dir})
        print(
            f"{prefix} finished: avg_system={row['avg_system_goodput_mbps']:.3f} Mbps, "
            f"p05_system={row['p05_system_goodput_mbps']:.3f} Mbps, "
            f"avg_ue={row['avg_ue_goodput_mbps']:.3f} Mbps, "
            f"p05_ue={row['p05_ue_goodput_mbps']:.3f} Mbps, "
            f"actual_mcs[avg,p05]=[{row['avg_actual_mcs']:.3f}, {row['p05_actual_mcs']:.3f}], "
            "effective_sinr_db[p05,p50,p95]="
            f"[{row['p05_effective_sinr_db']:.3f}, {row['p50_effective_sinr_db']:.3f}, "
            f"{row['p95_effective_sinr_db']:.3f}]",
            flush=True,
        )

    write_csv(out_dir / "metrics" / "parameter_sweep_summary.csv", rows)
    write_json(out_dir / "metrics" / "parameter_sweep_summary.json", {
        "parameter": parameter,
        "values": values,
        "common_random_seed": base_seed,
        "runs": rows,
    })
    _combined_plots(out_dir, parameter, run_records)
    return rows
