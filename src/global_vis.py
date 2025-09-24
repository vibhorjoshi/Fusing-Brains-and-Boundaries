from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Approximate state centroids for a basic scatter map
STATE_COORDS = {
    "Alabama": (32.8, -86.8), "Arizona": (33.7, -111.4), "Arkansas": (35.0, -92.4), "California": (36.1, -119.7),
    "Colorado": (39.1, -105.3), "Connecticut": (41.6, -72.8), "Delaware": (39.3, -75.5), "Florida": (27.8, -81.7),
    "Georgia": (33.0, -83.6), "Idaho": (44.2, -114.5), "Illinois": (40.3, -89.0), "Indiana": (39.8, -86.3),
    "Iowa": (42.0, -93.2), "Kansas": (38.5, -96.7), "Kentucky": (37.7, -84.7), "Louisiana": (31.2, -91.9),
    "Maine": (44.7, -69.4), "Maryland": (39.1, -76.8), "Massachusetts": (42.2, -71.5), "Michigan": (43.3, -84.5),
    "Minnesota": (45.7, -93.9), "Mississippi": (32.7, -89.7), "Missouri": (38.5, -92.3), "Montana": (46.9, -110.5),
    "Nebraska": (41.1, -98.3), "Nevada": (38.3, -117.1), "NewHampshire": (43.5, -71.6), "NewJersey": (40.3, -74.5),
    "NewMexico": (34.8, -106.2), "NewYork": (42.2, -74.9), "NorthCarolina": (35.6, -79.8), "NorthDakota": (47.5, -99.8),
    "Ohio": (40.4, -82.8), "Oklahoma": (35.6, -96.9), "Oregon": (44.6, -122.1), "Pennsylvania": (40.6, -77.2),
    "RhodeIsland": (41.7, -71.5), "SouthCarolina": (33.9, -80.9), "SouthDakota": (44.3, -99.4), "Tennessee": (35.7, -86.7),
    "Texas": (31.1, -97.6), "Utah": (40.2, -111.9), "Vermont": (44.0, -72.7), "Virginia": (37.8, -78.2),
    "Washington": (47.4, -121.5), "WestVirginia": (38.5, -81.0), "Wisconsin": (44.3, -89.6), "Wyoming": (42.8, -107.3),
}


def plot_us_state_bars(csv_path: Path, fig_path: Path):
    df = pd.read_csv(csv_path)
    # prefer mean_iou if present, fall back to rl_iou, else improvement
    value_col = None
    for c in ["mean_iou", "rl_iou", "improvement"]:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        raise ValueError("CSV missing expected columns: mean_iou/rl_iou/improvement")
    df = df.sort_values(value_col, ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(df["state"], df[value_col], color="#4daf4a")
    plt.xticks(rotation=90)
    ylabel = "Value"
    if value_col == "mean_iou":
        ylabel = "Mean IoU (RL)"
    elif value_col == "rl_iou":
        ylabel = "RL IoU"
    elif value_col == "improvement":
        ylabel = "IoU Improvement"
    plt.ylabel(ylabel)
    plt.title("Per-State {} (Top {} States)".format(ylabel, len(df)))
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.close()


def plot_3d_bars(csv_path: Path, fig_path: Path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    df = pd.read_csv(csv_path)
    # choose column
    z_col = "mean_iou" if "mean_iou" in df.columns else ("improvement" if "improvement" in df.columns else "rl_iou")
    x = np.arange(len(df))
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    dx = np.ones_like(x) * 0.6
    dy = np.ones_like(x) * 0.6
    dz = df[z_col].to_numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(x, y, z, dx, dy, dz, color="#377eb8", shade=True)
    ax.set_zlim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["state"], rotation=90)
    ax.set_zlabel(z_col)
    ax.set_title("3D View: {} by State".format(z_col))
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_global_visualizations(csv_path: Path, out_dir: Path):
    plot_us_state_bars(csv_path, out_dir / "multistate_iou_bars.png")
    plot_3d_bars(csv_path, out_dir / "multistate_iou_3d.png")


def plot_us_improvement_map(summary_table: List[dict], fig_path: Path):
    lats, lons, sizes, labels = [], [], [], []
    for row in summary_table:
        st = row["state"]
        imp = float(row.get("improvement", row.get("rl_iou", 0.0) - row.get("baseline_iou", 0.0)))
        coord = STATE_COORDS.get(st)
        if not coord:
            continue
        lat, lon = coord
        lats.append(lat)
        lons.append(lon)
        sizes.append(max(20, 400 * abs(imp)))
        labels.append(f"{st}: {imp:.3f}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("RL Fusion IoU Improvement (by State)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    sc = ax.scatter(lons, lats, s=sizes, c=np.clip(np.array(sizes)/400.0, 0, 1), cmap="YlGn")
    for (x, y, lab) in zip(lons, lats, labels):
        ax.text(x, y, lab, fontsize=8)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
