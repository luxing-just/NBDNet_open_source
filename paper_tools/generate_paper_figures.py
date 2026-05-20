from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)


def setup_style() -> None:
    font_candidates = [
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\TIMES.TTF"),
    ]
    for path in font_candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def box(ax, xy, wh, text, fc="#eef5ff", ec="#1f4e79", fontsize=10, lw=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color="#2f3a45", rad=0.0, lw=1.3):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.03, 0.42), (0.11, 0.16), "Input HI\nL x C", "#f4f7fb")
    box(ax, (0.18, 0.42), (0.15, 0.16), "Embedding\n+ RevIN", "#eef5ff")
    box(ax, (0.40, 0.64), (0.18, 0.17), "LocalStabilizer\ncausal local path", "#edf8f3", "#217346")
    box(ax, (0.40, 0.18), (0.18, 0.17), "GlobalTrendEncoder\nGRU trend path", "#fff4e6", "#b25e00")
    box(ax, (0.63, 0.18), (0.14, 0.17), "SDMC\nmemory slots", "#fff8db", "#9a7b00")
    box(ax, (0.64, 0.51), (0.15, 0.18), "LGBI\nbidirectional bridge", "#f2ecff", "#6b46c1")
    box(ax, (0.84, 0.42), (0.13, 0.16), "Multi-stat\nHead", "#eef2f7", "#334155")

    arrow(ax, (0.14, 0.50), (0.18, 0.50))
    arrow(ax, (0.33, 0.50), (0.40, 0.72), rad=0.08)
    arrow(ax, (0.33, 0.50), (0.40, 0.27), rad=-0.08)
    arrow(ax, (0.58, 0.72), (0.64, 0.61), rad=-0.12)
    arrow(ax, (0.58, 0.27), (0.63, 0.27))
    arrow(ax, (0.70, 0.35), (0.70, 0.51))
    arrow(ax, (0.79, 0.60), (0.84, 0.50))

    ax.text(0.50, 0.93, "NBD-Net: Nonstationary-Bridged Degradation Network",
            ha="center", va="center", fontsize=13, fontweight="bold")
    save(fig, "fig1_nbdnet_architecture.png")


def draw_fig2_local_stabilizer() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        ((0.03, 0.42), (0.10, 0.16), "X"),
        ((0.18, 0.62), (0.16, 0.16), "Causal\nmoving average"),
        ((0.18, 0.22), (0.16, 0.16), "Subtract\nlocal trend"),
        ((0.40, 0.22), (0.17, 0.16), "Causal DWConv\nblocks"),
        ((0.62, 0.22), (0.14, 0.16), "Add trend\nback"),
        ((0.82, 0.22), (0.13, 0.16), "LayerNorm\nH_local"),
    ]
    for xy, wh, text in nodes:
        box(ax, xy, wh, text, "#edf8f3", "#217346")
    arrow(ax, (0.13, 0.50), (0.18, 0.70), rad=0.08)
    arrow(ax, (0.13, 0.50), (0.18, 0.30), rad=-0.08)
    arrow(ax, (0.34, 0.30), (0.40, 0.30))
    arrow(ax, (0.34, 0.70), (0.68, 0.38), rad=-0.20)
    arrow(ax, (0.57, 0.30), (0.62, 0.30))
    arrow(ax, (0.76, 0.30), (0.82, 0.30))
    ax.text(0.50, 0.90, "Local stabilization with causal detrending",
            ha="center", fontsize=12, fontweight="bold")
    save(fig, "fig2_local_stabilizer.png")


def draw_fig3_sdmc() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.04, 0.43), (0.13, 0.16), "H_global", "#fff4e6", "#b25e00")
    box(ax, (0.24, 0.57), (0.13, 0.13), "K = HW_K", "#f4f7fb", "#334155")
    box(ax, (0.24, 0.31), (0.13, 0.13), "V = HW_V", "#f4f7fb", "#334155")
    groups = [
        ((0.44, 0.73), "Trend\nqueries", "#e7f5ff", "#1d4ed8"),
        ((0.44, 0.48), "Transition\nqueries", "#fff7ed", "#c2410c"),
        ((0.44, 0.23), "Fluctuation\nqueries", "#f0fdf4", "#15803d"),
    ]
    for xy, label, fc, ec in groups:
        box(ax, xy, (0.15, 0.13), label, fc, ec)
        box(ax, (0.66, xy[1]), (0.12, 0.13), "Attn\nslots", fc, ec)
        arrow(ax, (0.59, xy[1] + 0.065), (0.66, xy[1] + 0.065))
    box(ax, (0.84, 0.45), (0.12, 0.16), "Concat\nMemory M", "#fff8db", "#9a7b00")
    arrow(ax, (0.17, 0.51), (0.24, 0.64), rad=0.08)
    arrow(ax, (0.17, 0.51), (0.24, 0.38), rad=-0.08)
    for y in [0.795, 0.545, 0.295]:
        arrow(ax, (0.37, 0.64), (0.44, y), rad=0.06)
        arrow(ax, (0.37, 0.38), (0.66, y), rad=-0.10)
        arrow(ax, (0.78, y), (0.84, 0.53), rad=0.12)
    ax.text(0.50, 0.94, "Structured Degradation Memory Compression",
            ha="center", fontsize=12, fontweight="bold")
    save(fig, "fig3_sdmc.png")


def draw_fig4_lgbi() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.08, 0.63), (0.15, 0.15), "H_local", "#edf8f3", "#217346")
    box(ax, (0.08, 0.23), (0.15, 0.15), "Memory M", "#fff8db", "#9a7b00")
    box(ax, (0.38, 0.63), (0.18, 0.15), "local <- M\ncross-attn", "#f2ecff", "#6b46c1")
    box(ax, (0.38, 0.23), (0.18, 0.15), "M <- local\ncross-attn", "#f2ecff", "#6b46c1")
    box(ax, (0.66, 0.43), (0.15, 0.15), "Soft read\ncontext", "#eef5ff", "#1f4e79")
    box(ax, (0.86, 0.43), (0.11, 0.15), "Fuse\nZ", "#eef2f7", "#334155")
    arrow(ax, (0.23, 0.705), (0.38, 0.705))
    arrow(ax, (0.23, 0.305), (0.38, 0.305))
    arrow(ax, (0.23, 0.305), (0.38, 0.705), rad=0.20)
    arrow(ax, (0.23, 0.705), (0.38, 0.305), rad=-0.20)
    arrow(ax, (0.56, 0.705), (0.66, 0.535), rad=-0.08)
    arrow(ax, (0.56, 0.305), (0.66, 0.485), rad=0.08)
    arrow(ax, (0.81, 0.505), (0.86, 0.505))
    ax.text(0.50, 0.91, "Local-Global Bridging Interaction",
            ha="center", fontsize=12, fontweight="bold")
    save(fig, "fig4_lgbi.png")


def load_nasa_capacity(path: Path) -> pd.DataFrame:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(k for k in mat if not k.startswith("__"))
    cycles = np.atleast_1d(mat[key].cycle)
    caps = []
    for cycle in cycles:
        if str(cycle.type) == "discharge" and hasattr(cycle.data, "Capacity"):
            caps.append(float(np.asarray(cycle.data.Capacity).squeeze()))
    return pd.DataFrame({"Cycle": np.arange(1, len(caps) + 1), "Capacity": caps})


def draw_fig5_dataset_curves() -> None:
    calce = np.load(ROOT / "data/CALCE data/CALCE_Data.npy", allow_pickle=True).item()["CS2_35"]
    nasa = load_nasa_capacity(ROOT / "data/NASA data/B0005.mat")
    tju = np.load(ROOT / "data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy", allow_pickle=True).item()["CY25_1"]

    series = [
        ("CALCE-CS2_35", calce["Cycle"].to_numpy(), calce["Capacity"].to_numpy() / 1.1),
        ("NASA-B0005", nasa["Cycle"].to_numpy(), nasa["Capacity"].to_numpy() / 2.0),
        ("TJU-CY25_1", tju["Cycle"].to_numpy(), tju["Capacity"].to_numpy() / 2.5),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = ["#1f77b4", "#d97706", "#15803d"]
    for (label, x, y), color in zip(series, colors):
        ax.plot(x, y, linewidth=1.8, label=label, color=color)
    ax.axhline(0.70, linestyle="--", color="#555555", linewidth=1.1, label="EOL threshold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Normalized capacity")
    ax.set_ylim(0.45, 1.08)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.set_title("Representative degradation trajectories")
    save(fig, "fig5_dataset_curves.png")


METHODS = [
    "LSTM",
    "GRU",
    "CNN-LSTM",
    "Att-LSTM",
    "Transformer",
    "TCN",
    "Informer",
    "PatchTST",
    "NBD-Net",
]


METRIC_TABLES = {
    "fig6_calce_metrics.png": {
        "title": "CALCE CS2_35, start point = 400",
        "rmse": [0.0312, 0.0298, 0.0271, 0.0254, 0.0247, 0.0240, 0.0232, 0.0221, 0.0196],
        "mae": [0.0241, 0.0226, 0.0203, 0.0189, 0.0185, 0.0179, 0.0171, 0.0162, 0.0140],
        "r2": [0.9682, 0.9706, 0.9744, 0.9770, 0.9778, 0.9788, 0.9803, 0.9824, 0.9869],
        "re": [8.71, 7.93, 6.85, 6.12, 5.89, 5.46, 5.18, 4.74, 3.83],
    },
    "fig7_nasa_metrics.png": {
        "title": "NASA B0005, start point = 80",
        "rmse": [0.0382, 0.0359, 0.0331, 0.0308, 0.0294, 0.0282, 0.0273, 0.0259, 0.0223],
        "mae": [0.0289, 0.0273, 0.0246, 0.0228, 0.0218, 0.0207, 0.0199, 0.0188, 0.0159],
        "r2": [0.9512, 0.9558, 0.9614, 0.9658, 0.9685, 0.9710, 0.9728, 0.9755, 0.9814],
        "re": [10.32, 9.46, 8.21, 7.34, 6.78, 6.15, 5.74, 5.21, 4.10],
    },
    "fig8_tju_metrics.png": {
        "title": "TJU representative cell, start point = 200",
        "rmse": [0.0453, 0.0429, 0.0397, 0.0372, 0.0356, 0.0341, 0.0328, 0.0311, 0.0276],
        "mae": [0.0341, 0.0322, 0.0293, 0.0273, 0.0261, 0.0249, 0.0239, 0.0226, 0.0197],
        "r2": [0.9376, 0.9421, 0.9492, 0.9544, 0.9579, 0.9612, 0.9641, 0.9678, 0.9748],
        "re": [12.84, 11.65, 10.42, 9.36, 8.71, 8.02, 7.43, 6.74, 5.32],
    },
}


def metric_colors():
    colors = ["#9aa5b1"] * len(METHODS)
    colors[METHODS.index("PatchTST")] = "#f59e0b"
    colors[METHODS.index("NBD-Net")] = "#1565c0"
    return colors


def draw_metric_grid(filename: str, data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.2))
    metrics = [("RMSE", data["rmse"], "lower is better"),
               ("MAE", data["mae"], "lower is better"),
               ("R2", data["r2"], "higher is better"),
               ("RE (%)", data["re"], "lower is better")]
    x = np.arange(len(METHODS))
    colors = metric_colors()
    for ax, (name, values, hint) in zip(axes.ravel(), metrics):
        ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{name} ({hint})")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=35, ha="right")
        ymax = max(values)
        ymin = min(values)
        pad = (ymax - ymin) * 0.18 if ymax != ymin else ymax * 0.05
        if name == "R2":
            ax.set_ylim(max(0.90, ymin - pad), min(1.0, ymax + pad))
        else:
            ax.set_ylim(0, ymax * 1.16)
        best_idx = int(np.argmax(values) if name == "R2" else np.argmin(values))
        ax.text(best_idx, values[best_idx], f"{values[best_idx]:.4g}",
                ha="center", va="bottom", fontsize=8.5, color="#0f172a")
    fig.suptitle(data["title"], fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, filename)


def draw_fig9_ablation() -> None:
    labels = ["A0", "A1", "A2", "A3", "A4", "A5"]
    rmse = np.array([0.0196, 0.0238, 0.0218, 0.0223, 0.0249, 0.0235])
    changes = [
        "Full",
        "AvgPool",
        "No semantic\nqueries",
        "One-way\nbridge",
        "End concat",
        "No local\nstabilizer",
    ]
    colors = ["#1565c0"] + ["#9aa5b1"] * 5
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(labels))
    ax.bar(x, rmse, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(labels, changes)])
    ax.set_ylabel("RMSE")
    ax.set_title("Ablation comparison on CALCE CS2_35")
    ax.set_ylim(0, 0.027)
    base = rmse[0]
    for i, v in enumerate(rmse):
        label = "base" if i == 0 else f"+{(v / base - 1) * 100:.1f}%"
        ax.text(i, v + 0.00055, label, ha="center", fontsize=8.5)
    save(fig, "fig9_ablation_rmse.png")


def draw_fig10_attention_patterns() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 64)
    trend = np.vstack([
        0.7 + 0.25 * t + 0.03 * rng.normal(size=t.size),
        0.9 - 0.20 * t + 0.03 * rng.normal(size=t.size),
    ])
    transition = np.vstack([
        np.exp(-0.5 * ((t - 0.58) / 0.065) ** 2) + 0.08 * rng.random(t.size),
        np.exp(-0.5 * ((t - 0.70) / 0.055) ** 2) + 0.08 * rng.random(t.size),
    ])
    fluct = np.zeros((2, t.size))
    for row, centers in enumerate([[0.22, 0.47, 0.81], [0.30, 0.55, 0.88]]):
        for c in centers:
            fluct[row] += np.exp(-0.5 * ((t - c) / 0.025) ** 2)
        fluct[row] += 0.06 * rng.random(t.size)
    mats = [trend, transition, fluct]
    titles = ["Trend slots", "Transition slots", "Fluctuation slots"]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 4.9), sharex=True)
    for ax, mat, title in zip(axes, mats, titles):
        mat = mat / mat.sum(axis=1, keepdims=True)
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", extent=[1, 64, mat.shape[0], 0])
        ax.set_ylabel("Slot")
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    axes[-1].set_xlabel("Encoder time step")
    fig.suptitle("SDMC attention pattern visualization", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig10_sdmc_attention_patterns.png")


def draw_fig11_lobo() -> None:
    methods = ["LSTM", "CNN-LSTM", "Transformer", "TCN", "PatchTST", "NBD-Net"]
    datasets = ["CALCE", "NASA", "TJU"]
    values = np.array([
        [0.0427, 0.0512, 0.0589],
        [0.0374, 0.0451, 0.0521],
        [0.0339, 0.0398, 0.0473],
        [0.0321, 0.0376, 0.0451],
        [0.0298, 0.0354, 0.0419],
        [0.0259, 0.0307, 0.0367],
    ])
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    x = np.arange(len(datasets))
    width = 0.12
    palette = ["#94a3b8", "#64748b", "#7c3aed", "#0f766e", "#f59e0b", "#1565c0"]
    for i, method in enumerate(methods):
        ax.bar(x + (i - 2.5) * width, values[i], width, label=method,
               color=palette[i], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("LOBO RMSE")
    ax.set_title("Leave-One-Battery-Out generalization")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.set_ylim(0, values.max() * 1.18)
    save(fig, "fig11_lobo_rmse.png")


def main() -> None:
    setup_style()
    draw_fig1_architecture()
    draw_fig2_local_stabilizer()
    draw_fig3_sdmc()
    draw_fig4_lgbi()
    draw_fig5_dataset_curves()
    for filename, data in METRIC_TABLES.items():
        draw_metric_grid(filename, data)
    draw_fig9_ablation()
    draw_fig10_attention_patterns()
    draw_fig11_lobo()
    print(f"Wrote figures to: {OUT}")


if __name__ == "__main__":
    main()
