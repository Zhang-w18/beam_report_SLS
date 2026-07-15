import csv
from types import SimpleNamespace

from scripts.plot_cdf import BASELINE_NO_INTERFERENCE, draw_cdf


def test_offline_cdf_can_overlay_baseline_no_interference(tmp_path):
    run_dir = tmp_path / "run"
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True)
    with (metrics / "ue_goodput.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["scheme", "drop", "ue_id", "avg_goodput_mbps", "scheduled"],
        )
        writer.writeheader()
        writer.writerows([
            {"scheme": "baseline", "drop": 0, "ue_id": 0,
             "avg_goodput_mbps": 10.0, "scheduled": 1},
            {"scheme": "baseline", "drop": 0, "ue_id": 1,
             "avg_goodput_mbps": 20.0, "scheduled": 1},
            {"scheme": BASELINE_NO_INTERFERENCE, "drop": 0, "ue_id": 0,
             "avg_goodput_mbps": 15.0, "scheduled": 1},
            {"scheme": BASELINE_NO_INTERFERENCE, "drop": 0, "ue_id": 1,
             "avg_goodput_mbps": 25.0, "scheduled": 1},
        ])

    output = run_dir / "figures" / "selected.pdf"
    export_data = metrics / "selected_ecdf.csv"
    args = SimpleNamespace(
        metric="ue_goodput",
        schemes=["baseline", BASELINE_NO_INTERFERENCE],
        font_size=13.0,
        title=None,
        xlabel=None,
        output=str(output),
        export_data=str(export_data),
        list_schemes=False,
    )
    config = {
        "curves": [
            {"scheme": "baseline", "color": "#d62728", "linestyle": "--"},
            {"scheme": BASELINE_NO_INTERFERENCE, "color": "#2ca02c", "linestyle": "-."},
        ]
    }

    figure_path, data_path = draw_cdf(run_dir, config, args)

    assert figure_path == output
    assert figure_path.stat().st_size > 0
    assert data_path == export_data
    text = export_data.read_text(encoding="utf-8")
    assert "Baseline SU (no interference)" in text
    assert f"{BASELINE_NO_INTERFERENCE}," in text
