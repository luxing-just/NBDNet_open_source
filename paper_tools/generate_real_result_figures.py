from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper_tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

import generate_paper_figures as base
import run_benchmark_experiments as bench


OUT = ROOT / "paper_figures_real"
OUT.mkdir(exist_ok=True)
base.OUT = OUT

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

METHOD_LABELS = {
    "Attention-LSTM": "Att-LSTM",
}


def labels(methods: list[str]) -> list[str]:
    return [METHOD_LABELS.get(m, m) for m in methods]


def colors(methods: list[str]) -> list[str]:
    palette = {
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
    return [palette[m] for m in methods]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


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
    bar_colors = colors(df["model"].tolist())
    for ax, (name, mean_col, std_col, hint, fmt) in zip(axes.ravel(), metric_defs):
        vals = df[mean_col].to_numpy(float)
        errs = df[std_col].fillna(0).to_numpy(float)
        ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=2.5,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        ax.set_title(f"{name} ({hint})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels(df["model"].tolist()), rotation=35, ha="right")
        best_idx = int(np.argmax(vals) if name == "R2" else np.argmin(vals))
        ymax = float(np.nanmax(vals + errs))
        ymin = float(np.nanmin(vals - errs))
        pad = (ymax - ymin) * 0.18 if ymax > ymin else max(abs(ymax) * 0.08, 0.01)
        if name == "R2":
            ax.set_ylim(max(-0.1, ymin - pad), min(1.02, ymax + pad))
        else:
            ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
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
    save(fig, filename)


def draw_all_metric_figures() -> None:
    summary = pd.read_csv(ROOT / "paper_experiment_results" / "benchmark_summary.csv")
    specs = [
        ("fig6_calce_metrics.png", "CALCE CS2_35, start point = 400", "CALCE"),
        ("fig7_nasa_metrics.png", "NASA B0005, start point = 80", "NASA"),
        ("fig8_tju_metrics.png", "TJU CY25_1, start point = 200", "TJU"),
    ]
    for filename, title, dataset in specs:
        draw_metric_grid(filename, title, summary[summary["dataset"] == dataset].copy())


def draw_ablation() -> None:
    df = pd.read_csv(ROOT / "paper_experiment_results" / "ablation_summary.csv")
    order = ["A0 Full", "A1 AvgPool", "A2 Unstructured", "A3 OneWay", "A4 EndConcat", "A5 NoLocal"]
    names = ["A0\nFull", "A1\nAvgPool", "A2\nUnstruct.", "A3\nOne-way", "A4\nConcat", "A5\nNo local"]
    df = df.set_index("variant").loc[order].reset_index()
    vals = df["RMSE_mean"].to_numpy(float)
    errs = df["RMSE_std"].fillna(0).to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.3, 3.85))
    x = np.arange(len(order))
    ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=2.5,
        color=["#1565c0"] + ["#9aa5b1"] * (len(order) - 1),
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("RMSE")
    ax.set_title("Ablation comparison on CALCE CS2_35")
    ax.set_ylim(0, float(np.max(vals + errs)) * 1.18)
    base = vals[0]
    for i, v in enumerate(vals):
        text = "base" if i == 0 else f"{(v / base - 1) * 100:+.2f}%"
        ax.text(i, v + errs[i] + ax.get_ylim()[1] * 0.015, text, ha="center", fontsize=8.3)
    save(fig, "fig9_ablation_rmse.png")


def sdmc_attention(model: bench.NBDNetRegressor, x: np.ndarray, device) -> list[np.ndarray]:
    net = model.net
    net.eval()
    xb = x[:1].to(device)
    with __import__("torch").no_grad():
        h = net.input_embedding(xb)
        h = net.revin(h, mode="norm")
        h_global = net.global_enc(h)
        sdmc = net.sdmc
        k = sdmc.W_k(h_global)
        groups = [sdmc.q_trend, sdmc.q_transition, sdmc.q_fluct]
        out = []
        for q in groups:
            b, _l, d = k.shape
            q_b = q.unsqueeze(0).expand(b, q.shape[0], d)
            q_p = sdmc.W_q(q_b)
            scores = __import__("torch").matmul(q_p, k.transpose(-2, -1)) / np.sqrt(d)
            out.append(__import__("torch").softmax(scores, dim=-1).squeeze(0).detach().cpu().numpy())
    return out


def draw_attention_patterns() -> None:
    import torch

    args = type("Args", (), {
        "seq_len": 64,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 80,
        "patience": 12,
        "seeds": [1],
    })()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series = bench.load_calce()
    train_ds, valid_ds = bench.build_train_valid([v for k, v in series.items() if k != "CS2_35"], args.seq_len)
    x_test, _y_true, _cycles = bench.build_test(series["CS2_35"], args.seq_len, 400)
    model = bench.train_one("NBD-Net", train_ds, valid_ds, args.seq_len, 1, args, device)
    mats = sdmc_attention(model, x_test, device)
    titles = ["Trend slots", "Transition slots", "Fluctuation slots"]
    rows = []
    for group, mat in zip(["trend", "transition", "fluctuation"], mats):
        for slot_idx, row in enumerate(mat, start=1):
            rows.append({"group": group, "slot": slot_idx, **{f"t{i+1}": v for i, v in enumerate(row)}})
    pd.DataFrame(rows).to_csv(OUT / "fig10_sdmc_attention_seed1.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 4.9), sharex=True)
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", extent=[1, mat.shape[1], mat.shape[0], 0])
        ax.set_ylabel("Slot")
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    axes[-1].set_xlabel("Encoder time step")
    fig.suptitle("SDMC attention patterns learned on CALCE CS2_35", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig10_sdmc_attention_patterns.png")


def draw_lobo() -> None:
    df = pd.read_csv(ROOT / "paper_experiment_results" / "lobo_summary.csv")
    methods = METHODS
    datasets = ["CALCE", "NASA", "TJU"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(datasets))
    width = 0.082
    for i, method in enumerate(methods):
        sub = df[df["model"] == method].set_index("dataset").loc[datasets]
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            sub["mean"].to_numpy(float),
            width,
            yerr=sub["std"].fillna(0).to_numpy(float),
            capsize=1.8,
            label=METHOD_LABELS.get(method, method),
            color=colors([method])[0],
            edgecolor="white",
            linewidth=0.45,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("LOBO RMSE")
    ax.set_title("Leave-One-Battery-Out generalization")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.set_ylim(0, float(df["mean"].max() + df["std"].max()) * 1.20)
    save(fig, "fig11_lobo_rmse.png")


def main() -> None:
    base.setup_style()
    base.draw_fig1_architecture()
    base.draw_fig2_local_stabilizer()
    base.draw_fig3_sdmc()
    base.draw_fig4_lgbi()
    base.draw_fig5_dataset_curves()
    draw_all_metric_figures()
    draw_ablation()
    draw_attention_patterns()
    draw_lobo()
    print(f"Wrote figures to: {OUT}")


if __name__ == "__main__":
    main()
