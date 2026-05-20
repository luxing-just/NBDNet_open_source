from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_tools"))

import run_specified_baseline_protocol as protocol

OUT_DIR = ROOT / "paper_experiment_results" / "robustness_analysis"
FIG_DIR = ROOT / "paper_figures_real" / "robustness_analysis"
BASE_DIR = ROOT / "paper_experiment_results" / "specified_baseline_protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
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

COLORS = {
    "LSTM": "#ff7f0e",
    "GRU": "#4c78a8",
    "CNN-LSTM": "#e45756",
    "Attention-LSTM": "#f58518",
    "Transformer": "#72b7b2",
    "TCN": "#54a24b",
    "Informer": "#9d755d",
    "PatchTST": "#b279a2",
    "NBD-Net": "#1f77b4",
}


def setup_matplotlib():
    import matplotlib as mpl
    from matplotlib import font_manager

    font = "Times New Roman"
    if font not in {f.name for f in font_manager.fontManager.ttflist}:
        font = "serif"
    mpl.rcParams["font.family"] = font
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.linewidth"] = 0.9


def load_raw_metrics() -> pd.DataFrame:
    raw_path = BASE_DIR / "specified_protocol_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path)
    raw["dataset_label"] = raw["dataset"].replace({"CALCE2": "CALCE-CS2_37"})
    raw.loc[raw["scenario"] == "calce", "dataset_label"] = "CALCE-CS2_35"
    raw.loc[raw["scenario"] == "nasa", "dataset_label"] = "NASA-B0005"
    raw.loc[raw["scenario"] == "tju", "dataset_label"] = "TJU-CY25_1"
    return raw


def plot_rmse_distribution(raw: pd.DataFrame) -> Path:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    labels = ["CALCE-CS2_35", "CALCE-CS2_37", "NASA-B0005", "TJU-CY25_1"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharey=False)
    axes = axes.ravel()
    rng = np.random.default_rng(2026)

    for ax, label in zip(axes, labels):
        sub = raw[raw["dataset_label"] == label]
        data = [sub[sub["model"] == model]["RMSE"].to_numpy(float) for model in MODEL_ORDER]
        positions = np.arange(1, len(MODEL_ORDER) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markersize": 3.4, "markerfacecolor": "white", "markeredgecolor": "#222222"},
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#444444", "linewidth": 0.9},
            capprops={"color": "#444444", "linewidth": 0.9},
            flierprops={"marker": "o", "markersize": 2.8, "markerfacecolor": "white", "markeredgecolor": "#777777"},
        )
        for patch, model in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.38)
            patch.set_edgecolor(COLORS[model])
            patch.set_linewidth(1.1)
        for pos, model, values in zip(positions, MODEL_ORDER, data):
            if values.size == 0:
                continue
            jitter = rng.normal(0, 0.045, size=values.size)
            ax.scatter(
                np.full(values.size, pos) + jitter,
                values,
                s=12,
                color=COLORS[model],
                alpha=0.72,
                edgecolors="none",
                zorder=3,
            )
        ax.set_title(label, fontsize=13, fontweight="bold", pad=6)
        ax.set_xticks(positions)
        ax.set_xticklabels(MODEL_ORDER, rotation=32, ha="right", fontsize=8.7)
        ax.set_ylabel("RMSE (Ah)", fontsize=12)
        ax.tick_params(axis="y", labelsize=10.5, direction="in", right=True)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#d9d9d9", alpha=0.75)
        all_values = np.concatenate([values for values in data if values.size])
        lo, hi = np.percentile(all_values, [1, 99])
        pad = max((hi - lo) * 0.24, 0.002)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)

    fig.tight_layout()
    out = FIG_DIR / "fig12_rmse_run_distribution_box.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def collect_point_errors(raw: pd.DataFrame, max_per_group: int = 4500) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2026)
    for _, row in raw.iterrows():
        scenario = row["scenario"]
        battery = row["battery"]
        model = row["model"]
        sp = int(row["start_point"])
        exp = int(row["exp"])
        pred_path = BASE_DIR / scenario / battery / "predictions" / f"{model}_SP{sp}_exp{exp}.csv"
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        err = np.abs(df["Error"].to_numpy(float))
        if err.size > max_per_group:
            idx = rng.choice(err.size, size=max_per_group, replace=False)
            err = err[idx]
        dataset_label = row["dataset_label"]
        rows.extend({"dataset_label": dataset_label, "model": model, "abs_error": float(v)} for v in err)
    return pd.DataFrame(rows)


def plot_point_error_violin(point_errors: pd.DataFrame) -> Path:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    labels = ["CALCE-CS2_35", "CALCE-CS2_37", "NASA-B0005", "TJU-CY25_1"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharey=False)
    axes = axes.ravel()

    for ax, label in zip(axes, labels):
        sub = point_errors[point_errors["dataset_label"] == label]
        data = [sub[sub["model"] == model]["abs_error"].to_numpy(float) for model in MODEL_ORDER]
        positions = np.arange(1, len(MODEL_ORDER) + 1)
        parts = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showmedians=True, showextrema=False)
        for body, model in zip(parts["bodies"], MODEL_ORDER):
            body.set_facecolor(COLORS[model])
            body.set_edgecolor(COLORS[model])
            body.set_alpha(0.35)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#111111")
            parts["cmedians"].set_linewidth(1.0)
        for pos, values, model in zip(positions, data, MODEL_ORDER):
            if values.size == 0:
                continue
            q1, q3 = np.percentile(values, [25, 75])
            ax.vlines(pos, q1, q3, color=COLORS[model], linewidth=3.0, alpha=0.90)
            ax.scatter(pos, np.median(values), s=15, color="#111111", zorder=3)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=6)
        ax.set_xticks(positions)
        ax.set_xticklabels(MODEL_ORDER, rotation=32, ha="right", fontsize=8.7)
        ax.set_ylabel("Absolute error (Ah)", fontsize=12)
        ax.tick_params(axis="y", labelsize=10.5, direction="in", right=True)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#d9d9d9", alpha=0.75)
        hi = np.percentile(sub["abs_error"].to_numpy(float), 99.2)
        ax.set_ylim(0, max(hi * 1.12, 1e-3))

    fig.tight_layout()
    out = FIG_DIR / "fig13_point_error_distribution_violin.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_datasets(module, scenario, args, battery_data, start_point, noise_level: float, seed: int):
    df_train, df_test, df_all = module.battery_data_process(battery_data, scenario.test_name, start_point, args)
    clean_test = df_test.copy()
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, noise_level, size=len(df_test))
        x = clean_test["target"].to_numpy(float)
        z = clean_test["Capacity"].to_numpy(float)
        slope = np.polyfit(x, z, 1)[0] if len(np.unique(x)) > 1 else 1.0
        df_test = df_test.copy()
        df_test["target"] = np.clip(x + eps, 0.0, 1.5)
        df_test["Capacity"] = np.clip(z + eps * slope, -0.25, 1.75)

    ds_kwargs = dict(
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        min_encoder_length=args.seq_len,
        max_encoder_length=args.seq_len,
        min_prediction_length=args.pred_len,
        max_prediction_length=args.pred_len,
        time_varying_known_reals=["Capacity"],
        time_varying_unknown_reals=["target"],
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False,
    )
    train_cut = int(0.8 * len(df_train))
    training_ds = TimeSeriesDataSet(df_train[:train_cut], **ds_kwargs)
    test_ds = TimeSeriesDataSet(df_test, **ds_kwargs)
    return training_ds, test_ds, df_all


def eval_noisy_checkpoint(model_name: str, module, scenario, args, battery_data, sp: int, exp: int,
                          noise_level: float, device: torch.device) -> dict:
    ckpt_path = BASE_DIR / scenario.key / scenario.test_name / "checkpoints" / model_name / f"SP{sp}_exp{exp}" / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    training_ds, test_ds, df_all = make_datasets(
        module, scenario, args, battery_data, sp, noise_level=noise_level, seed=10_000 + exp + int(noise_level * 10_000)
    )
    test_dl = test_ds.to_dataloader(
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = protocol.load_best_model(model_name, module, training_ds, args, str(ckpt_path), device)
    with torch.no_grad():
        preds = model.predict(test_dl, batch_size=256).cpu().numpy().reshape(-1)

    act_df = df_all.loc[df_all["Cycle"] >= sp, ["Cycle", "target"]]
    y_true = act_df["target"].to_numpy(float) * scenario.rated_capacity
    y_pred = preds * scenario.rated_capacity
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    threshold = scenario.rated_capacity * 0.7
    rul_r, rul_p, ae, re = module.rul_error(y_true, y_pred, threshold)
    row = {
        "scenario": scenario.key,
        "dataset": scenario.dataset,
        "battery": scenario.test_name,
        "start_point": sp,
        "exp": exp,
        "model": model_name,
        "noise_level": noise_level,
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": float(r2_score(y_true, y_pred)),
        "RUL_real": rul_r,
        "RUL_pred": rul_p,
        "AE": ae,
        "RE": re,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def summarize_noise(noise_raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["RMSE", "MAE", "R2", "AE", "RE"]
    return noise_raw.groupby(
        ["scenario", "dataset", "battery", "start_point", "model", "noise_level"], as_index=False
    ).agg(
        **{f"{m}_mean": (m, "mean") for m in metric_cols},
        **{f"{m}_std": (m, "std") for m in metric_cols},
        n_exp=("exp", "count"),
    )


def plot_noise_robustness(noise_summary: pd.DataFrame) -> Path:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    labels = [
        ("calce", "CALCE-CS2_35"),
        ("calce2", "CALCE-CS2_37"),
        ("nasa", "NASA-B0005"),
        ("tju", "TJU-CY25_1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.2), sharex=True)
    axes = axes.ravel()
    markers = ["o", "s", "D", "<", "v", "^", "P", "h", "X"]

    for ax, (scenario_key, title) in zip(axes, labels):
        sub = noise_summary[noise_summary["scenario"] == scenario_key]
        for marker, model in zip(markers, MODEL_ORDER):
            mdf = sub[sub["model"] == model].sort_values("noise_level")
            if mdf.empty:
                continue
            x = mdf["noise_level"].to_numpy(float) * 100
            y = mdf["RMSE_mean"].to_numpy(float)
            yerr = mdf["RMSE_std"].fillna(0).to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[model],
                marker=marker,
                markersize=4.5 if model != "NBD-Net" else 5.8,
                linewidth=1.35 if model != "NBD-Net" else 2.0,
                capsize=2.0,
                alpha=0.92,
                label="NBD-Net(ours)" if model == "NBD-Net" else model,
            )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
        ax.set_xlabel("Gaussian noise level (%)", fontsize=12)
        ax.set_ylabel("RMSE (Ah)", fontsize=12)
        ax.set_xticks([0, 1, 3, 5])
        ax.tick_params(labelsize=10.5, direction="in", top=True, right=True)
        ax.grid(True, linestyle="--", linewidth=0.55, color="#d9d9d9", alpha=0.72)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=5, fontsize=9.2, frameon=True, fancybox=False,
               bbox_to_anchor=(0.5, 1.015), edgecolor="#bfbfbf")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "fig14_noise_robustness_rmse.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def run_noise(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    scenarios = [protocol.SCENARIOS[key] for key in args.scenarios]
    for scenario in scenarios:
        module = importlib.import_module(scenario.module_name)
        scenario_args = protocol.get_args(scenario, argparse.Namespace(
            count=args.count,
            max_epochs=200,
            precision=args.precision,
        ))
        battery_data = protocol.load_battery_data(module, scenario)
        start_points = (scenario.default_plot_sp,) if args.representative_only else scenario.start_points
        for sp in start_points:
            for model_name in args.models:
                for exp in range(1, args.count + 1):
                    clean_raw = BASE_DIR / scenario.key / scenario.test_name / "predictions" / f"{model_name}_SP{sp}_exp{exp}.metric.csv"
                    if clean_raw.exists():
                        row0 = pd.read_csv(clean_raw).iloc[0].to_dict()
                        row0["noise_level"] = 0.0
                        rows.append(row0)
                    for noise_level in args.noise_levels:
                        print(f"[noise {noise_level:.2%}] {scenario.key} {scenario.test_name} {model_name} SP{sp} exp{exp}")
                        row = eval_noisy_checkpoint(model_name, module, scenario, scenario_args, battery_data, sp, exp, noise_level, device)
                        rows.append(row)
    noise_raw = pd.DataFrame(rows)
    noise_raw.to_csv(OUT_DIR / "noise_robustness_raw.csv", index=False)
    noise_summary = summarize_noise(noise_raw)
    noise_summary.to_csv(OUT_DIR / "noise_robustness_summary.csv", index=False)
    return noise_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=["calce", "calce2", "nasa", "tju"],
                        choices=list(protocol.SCENARIOS))
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    parser.add_argument("--representative-only", action="store_true")
    parser.add_argument("--precision", default="16-mixed")
    args = parser.parse_args()

    outputs = []
    if not args.skip_distribution:
        raw = load_raw_metrics()
        out1 = plot_rmse_distribution(raw)
        point_errors = collect_point_errors(raw)
        point_errors.to_csv(OUT_DIR / "point_error_distribution_raw.csv", index=False)
        out2 = plot_point_error_violin(point_errors)
        outputs.extend([out1, out2])
    if not args.skip_noise:
        noise_summary = run_noise(args)
        out3 = plot_noise_robustness(noise_summary)
        outputs.append(out3)

    print("\nGenerated outputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
