from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures_real"
SUMMARY = ROOT / "paper_experiment_results" / "lobo_all_metrics_summary.csv"

METHODS = [
    "LSTM",
    "GRU",
    "CNN-LSTM",
    "Attention-LSTM",
    "Transformer",
    "TCN",
    "Informer",
    "PatchTST",
    "NBD-Net",
]

LABELS = {"Attention-LSTM": "Att-LSTM"}
PALETTE = {
    "LSTM": "#94a3b8",
    "GRU": "#7f8ea3",
    "CNN-LSTM": "#6b7280",
    "Attention-LSTM": "#9f7aea",
    "Transformer": "#7c3aed",
    "TCN": "#0f766e",
    "Informer": "#0891b2",
    "PatchTST": "#f59e0b",
    "NBD-Net": "#1565c0",
}


def setup_style() -> None:
    for path in [Path(r"C:\Windows\Fonts\times.ttf"), Path(r"C:\Windows\Fonts\TIMES.TTF")]:
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


def draw_metric_grid(filename: str, title: str, df: pd.DataFrame) -> None:
    df = df.set_index("model").loc[METHODS].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.35))
    metric_defs = [
        ("RMSE", "RMSE_mean", "RMSE_std", "lower is better", "{:.4f}"),
        ("MAE", "MAE_mean", "MAE_std", "lower is better", "{:.4f}"),
        ("R2", "R2_mean", "R2_std", "higher is better", "{:.4f}"),
        ("RE (%)", "RE_mean", "RE_std", "lower is better", "{:.2f}"),
    ]
    x = np.arange(len(df))
    colors = [PALETTE[m] for m in df["model"]]
    for ax, (name, mean_col, std_col, hint, fmt) in zip(axes.ravel(), metric_defs):
        vals = df[mean_col].to_numpy(float)
        errs = df[std_col].fillna(0).to_numpy(float)
        ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=2.5,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        ax.set_title(f"{name} ({hint})")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(m, m) for m in df["model"]], rotation=35, ha="right")
        if name == "R2":
            ymin = max(0.80, float(np.nanmin(vals - errs)) - 0.01)
            ax.set_ylim(ymin, 1.001)
            best_idx = int(np.nanargmax(vals))
        else:
            ymax = float(np.nanmax(vals + errs))
            ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
            best_idx = int(np.nanargmin(vals))
        ax.text(
            best_idx,
            vals[best_idx] + errs[best_idx] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015,
            fmt.format(vals[best_idx]),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#0f172a",
        )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    setup_style()
    summary = pd.read_csv(SUMMARY)
    specs = [
        ("fig6_calce_metrics.png", "CALCE leave-one-battery-out comparison", "CALCE"),
        ("fig7_nasa_metrics.png", "NASA leave-one-battery-out comparison", "NASA"),
        ("fig8_tju_metrics.png", "TJU leave-one-battery-out comparison", "TJU"),
    ]
    for filename, title, dataset in specs:
        draw_metric_grid(filename, title, summary[summary["dataset"] == dataset].copy())
    print(f"Wrote LOBO metric figures to {OUT}")


if __name__ == "__main__":
    main()
