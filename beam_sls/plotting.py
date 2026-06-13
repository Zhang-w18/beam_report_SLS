from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .utils import ensure_dir


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_cdf(values_by_label: Dict[str, Sequence[float]],
             xlabel: str,
             title: str,
             path: str | Path) -> None:
    plt = _mpl()
    ensure_dir(Path(path).parent)
    fig = plt.figure(figsize=(7, 4.5))
    for label, vals in values_by_label.items():
        x = np.sort(np.asarray(vals, dtype=float))
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1) / float(x.size)
        plt.plot(x, y, label=label)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bar(values: Dict[str, float], ylabel: str, title: str, path: str | Path) -> None:
    plt = _mpl()
    ensure_dir(Path(path).parent)
    labels = list(values.keys())
    x = np.arange(len(labels))
    fig = plt.figure(figsize=(7, 4.5))
    plt.bar(x, [values[k] for k in labels])
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heatmap(x_grid: np.ndarray,
                 y_grid: np.ndarray,
                 power_dbm: np.ndarray,
                 path: str | Path,
                 title: str = "TRP coverage heatmap") -> None:
    plt = _mpl()
    ensure_dir(Path(path).parent)
    fig = plt.figure(figsize=(6.5, 5.5))
    extent = [float(np.min(x_grid)), float(np.max(x_grid)), float(np.min(y_grid)), float(np.max(y_grid))]
    im = plt.imshow(power_dbm, origin="lower", extent=extent, aspect="equal")
    plt.colorbar(im, label="Average received power [dBm]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_best_beam_heatmap(x_grid: np.ndarray,
                           y_grid: np.ndarray,
                           best_beam: np.ndarray,
                           path: str | Path,
                           title: str = "Best beam index heatmap") -> None:
    plt = _mpl()
    ensure_dir(Path(path).parent)
    fig = plt.figure(figsize=(6.5, 5.5))
    extent = [float(np.min(x_grid)), float(np.max(x_grid)), float(np.min(y_grid)), float(np.max(y_grid))]
    masked = np.ma.masked_where(best_beam < 0, best_beam)
    im = plt.imshow(masked, origin="lower", extent=extent, aspect="equal")
    plt.colorbar(im, label="Best beam global index")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_topology(topology, cfg: Dict, path: str | Path) -> None:
    """Draw sites, sectors, dropped UEs, and ISD annotation."""
    plt = _mpl()
    ensure_dir(Path(path).parent)
    topo_cfg = cfg.get("topology", {})
    max_d = float(cfg.get("coverage_heatmap", {}).get("max_distance_m", cfg.get("scenario", {}).get("max_ue_distance_m", 300.0)))
    isd = float(topo_cfg.get("isd_m", cfg.get("scenario", {}).get("isd_m", 500.0)))
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for site in topology.sites:
        ax.scatter([site.x_m], [site.y_m], marker="^", s=90, label=f"Site {site.site_id}")
        ax.text(site.x_m + 5, site.y_m + 5, f"Site {site.site_id}", fontsize=9)

    # Draw sector boresights and sector edge lines.
    radius = max_d * 0.9
    for sec in topology.sectors:
        site = topology.site_by_id(sec.site_id)
        az = np.deg2rad(sec.azimuth_deg)
        half = np.deg2rad(sec.width_deg / 2.0)
        ax.plot([site.x_m, site.x_m + radius * np.cos(az)],
                [site.y_m, site.y_m + radius * np.sin(az)],
                linewidth=1.8)
        for edge in (az - half, az + half):
            ax.plot([site.x_m, site.x_m + radius * np.cos(edge)],
                    [site.y_m, site.y_m + radius * np.sin(edge)],
                    linestyle="--", linewidth=0.9)
        tx = site.x_m + (radius * 0.58) * np.cos(az)
        ty = site.y_m + (radius * 0.58) * np.sin(az)
        ax.text(tx, ty, f"Sector {sec.sector_id}\ncell {sec.cell_id}\naz={sec.azimuth_deg:.0f}°",
                ha="center", va="center", fontsize=8)

    if topology.ues:
        xs = [u.x_m for u in topology.ues]
        ys = [u.y_m for u in topology.ues]
        cs = [u.serving_cell for u in topology.ues]
        sc = ax.scatter(xs, ys, c=cs, s=16, alpha=0.8, label="UE")
        try:
            fig.colorbar(sc, ax=ax, label="Serving cell")
        except Exception:
            pass

    # ISD marker: for a one-site topology, draw a ghost neighbor on +x to annotate distance.
    if isd > 0:
        ax.scatter([isd], [0.0], marker="^", s=50, alpha=0.35)
        ax.annotate("", xy=(isd, -0.12 * max_d), xytext=(0.0, -0.12 * max_d),
                    arrowprops=dict(arrowstyle="<->", linewidth=1.2))
        ax.text(isd / 2.0, -0.17 * max_d, f"ISD = {isd:.0f} m", ha="center", va="top", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Topology: sites, 3 sectors, UE drops, inter-site distance")
    lim = max(max_d, isd * 0.6)
    ax.set_xlim(-lim, max(isd * 1.05, lim))
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
