from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RUL_NBDNet import NBDNetModel

OUT_DIR = ROOT / "paper_experiment_results" / "specified_baseline_protocol"
FIG_DIR = ROOT / "paper_figures_real" / "specified_baseline_protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


BASELINE_MODELS = [
    "LSTM",
    "GRU",
    "CNN-LSTM",
    "Attention-LSTM",
    "Transformer",
    "TCN",
    "Informer",
    "PatchTST",
]
ALL_MODELS = BASELINE_MODELS + ["NBD-Net"]


@dataclass(frozen=True)
class Scenario:
    key: str
    dataset: str
    module_name: str
    test_name: str
    start_points: tuple[int, ...]
    rated_capacity: float
    seq_len: int
    batch_size: int
    data_path: str | None = None
    data_dir: str | None = None
    battery_list: tuple[str, ...] | None = None
    default_plot_sp: int | None = None
    panel_label: str = "(a)"


SCENARIOS = {
    "calce": Scenario(
        key="calce",
        dataset="CALCE",
        module_name="RUL_Baselines_CALCE_8baselines",
        test_name="CS2_35",
        start_points=(300, 400, 500),
        rated_capacity=1.1,
        seq_len=64,
        batch_size=128,
        data_path="data/CALCE data/CALCE_Data.npy",
        default_plot_sp=400,
        panel_label="(a)",
    ),
    "calce2": Scenario(
        key="calce2",
        dataset="CALCE2",
        module_name="RUL_Baselines_CALCE2_8baselines",
        test_name="CS2_37",
        start_points=(300, 400, 500),
        rated_capacity=1.1,
        seq_len=64,
        batch_size=128,
        data_path="data/CALCE data/CALCE_Data.npy",
        default_plot_sp=400,
        panel_label="(b)",
    ),
    "nasa": Scenario(
        key="nasa",
        dataset="NASA",
        module_name="RUL_Baselines_NASA_8baselines",
        test_name="B0005",
        start_points=(50, 60, 70),
        rated_capacity=2.0,
        seq_len=30,
        batch_size=16,
        data_dir="data/NASA data/",
        battery_list=("B0005", "B0006", "B0007", "B0018"),
        default_plot_sp=70,
        panel_label="(c)",
    ),
    "tju": Scenario(
        key="tju",
        dataset="TJU",
        module_name="RUL_Baselines_TJU_8baselines",
        test_name="CY25_1",
        start_points=(200, 300, 400),
        rated_capacity=2.5,
        seq_len=64,
        batch_size=128,
        data_path="data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy",
        default_plot_sp=300,
        panel_label="(d)",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_args(scenario: Scenario, cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=1,
        seq_len=scenario.seq_len,
        pred_len=1,
        dropout=0.1,
        count=cli_args.count,
        batch_size=scenario.batch_size,
        Rated_Capacity=scenario.rated_capacity,
        test_name=scenario.test_name,
        start_point_list=list(scenario.start_points),
        max_epochs=cli_args.max_epochs,
        precision=cli_args.precision,
        data_path=scenario.data_path,
        data_dir=scenario.data_dir,
        Battery_list=list(scenario.battery_list) if scenario.battery_list else None,
        d_model=64,
        n_local_layers=3,
        local_kernel=7,
        local_window=9,
        n_global_layers=2,
        n_trend_slots=2,
        n_transition_slots=2,
        n_fluct_slots=2,
    )


def load_battery_data(module, scenario: Scenario):
    if scenario.key == "nasa":
        return module.DataRead(list(scenario.battery_list), scenario.data_dir)
    return np.load(ROOT / scenario.data_path, allow_pickle=True).item()


def build_model(model_name: str, module, training_ds, args):
    if model_name != "NBD-Net":
        return module.build_model(model_name, training_ds, args)
    return NBDNetModel.from_dataset(
        training_ds,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=1,
        d_model=args.d_model,
        n_local_layers=args.n_local_layers,
        local_kernel=args.local_kernel,
        local_window=args.local_window,
        n_global_layers=args.n_global_layers,
        n_trend_slots=args.n_trend_slots,
        n_transition_slots=args.n_transition_slots,
        n_fluct_slots=args.n_fluct_slots,
        dropout=args.dropout,
        learning_rate=0.001,
        loss=SMAPE(),
    )


def load_best_model(model_name: str, module, training_ds, args, best_path: str, device: torch.device):
    model = build_model(model_name, module, training_ds, args)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    if model_name == "NBD-Net":
        for key in ["network.revin.mean", "network.revin.stdev"]:
            state.pop(key, None)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def metric_row(model_name: str, scenario: Scenario, start_point: int, exp_id: int, n_params: int,
               train_time: float, y_true: np.ndarray, y_pred: np.ndarray, module) -> dict:
    threshold = scenario.rated_capacity * 0.7
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    rul_r, rul_p, ae, re = module.rul_error(y_true, y_pred, threshold)
    return {
        "scenario": scenario.key,
        "dataset": scenario.dataset,
        "battery": scenario.test_name,
        "start_point": start_point,
        "exp": exp_id,
        "model": model_name,
        "n_params": n_params,
        "train_time_s": round(train_time, 3),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "RUL_real": rul_r,
        "RUL_pred": rul_p,
        "AE": ae,
        "RE": re,
    }


def prediction_path(scenario: Scenario, model_name: str, start_point: int, exp_id: int) -> Path:
    safe_model = model_name.replace("/", "_")
    return OUT_DIR / scenario.key / scenario.test_name / "predictions" / f"{safe_model}_SP{start_point}_exp{exp_id}.csv"


def run_one(model_name: str, module, scenario: Scenario, args, battery_data, start_point: int, exp_id: int,
            device: torch.device, force: bool = False) -> dict:
    pred_csv = prediction_path(scenario, model_name, start_point, exp_id)
    metric_csv = pred_csv.with_suffix(".metric.csv")
    if pred_csv.exists() and metric_csv.exists() and not force:
        return pd.read_csv(metric_csv).iloc[0].to_dict()

    set_seed(exp_id)
    df_train, df_test, df_all = module.battery_data_process(battery_data, scenario.test_name, start_point, args)
    mask_len = len(df_train)

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
    training_ds = TimeSeriesDataSet(df_train[: int(0.8 * mask_len)], **ds_kwargs)
    valid_ds = TimeSeriesDataSet(df_train[int(0.8 * mask_len):], **ds_kwargs)
    test_ds = TimeSeriesDataSet(df_test, **ds_kwargs)

    dl_kwargs = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    train_dl = training_ds.to_dataloader(train=True, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    val_dl = valid_ds.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, drop_last=False, **dl_kwargs)
    test_dl = test_ds.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, drop_last=False, **dl_kwargs)

    model = build_model(model_name, module, training_ds, args)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_dir = OUT_DIR / scenario.key / scenario.test_name / "checkpoints" / model_name / f"SP{start_point}_exp{exp_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename="best", monitor="val_loss", mode="min", save_top_k=1),
        ],
        enable_progress_bar=False,
        default_root_dir=str(ckpt_dir),
        deterministic=True,
        logger=False,
    )
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    train_time = time.time() - t0

    best_model = load_best_model(model_name, module, training_ds, args, trainer.checkpoint_callback.best_model_path, device)
    preds = best_model.predict(test_dl, batch_size=256).cpu().numpy().reshape(-1)

    act_df = df_all.loc[df_all["Cycle"] >= start_point, ["Cycle", "target"]]
    y_true = act_df["target"].to_numpy(float) * scenario.rated_capacity
    y_pred = preds * scenario.rated_capacity
    cycles = act_df["Cycle"].to_numpy()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred, cycles = y_true[:n], y_pred[:n], cycles[:n]

    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Cycle": cycles,
            "Real_Capacity": y_true,
            "Predicted_Capacity": y_pred,
            "Error": y_pred - y_true,
        }
    ).to_csv(pred_csv, index=False)

    row = metric_row(model_name, scenario, start_point, exp_id, n_params, train_time, y_true, y_pred, module)
    pd.DataFrame([row]).to_csv(metric_csv, index=False)

    del model, best_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["RMSE", "MAE", "R2", "AE", "RE", "train_time_s", "n_params"]
    summary_sp = raw.groupby(["scenario", "dataset", "battery", "start_point", "model"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metric_cols},
        **{f"{m}_std": (m, "std") for m in metric_cols},
        n_exp=("exp", "count"),
    )
    summary_overall = raw.groupby(["scenario", "dataset", "battery", "model"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metric_cols},
        **{f"{m}_std": (m, "std") for m in metric_cols},
        n_exp=("exp", "count"),
    )
    return summary_sp, summary_overall


def plot_capacity_error(scenario: Scenario, start_point: int, models: list[str]) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = "Times New Roman"
    if font not in {f.name for f in font_manager.fontManager.ttflist}:
        font = "serif"
    mpl.rcParams["font.family"] = font
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.linewidth"] = 0.95

    style = {
        "LSTM": ("#ff7f0e", "s", 3.6),
        "GRU": ("#4c78a8", "o", 3.6),
        "CNN-LSTM": ("#e45756", "D", 3.6),
        "Attention-LSTM": ("#f58518", "<", 3.6),
        "Transformer": ("#72b7b2", "v", 3.8),
        "TCN": ("#54a24b", "^", 3.8),
        "Informer": ("#9d755d", "P", 3.8),
        "PatchTST": ("#b279a2", "P", 3.8),
        "NBD-Net": ("#1f77b4", "X", 4.6),
    }
    real_color = "#4d4d4d"
    threshold_color = "#1a1a1a"
    sp_color = "#7a1fa2"
    grid_color = "#d9d9d9"
    threshold = scenario.rated_capacity * 0.7

    model_frames = {}
    real = None
    cycles = None
    for model in models:
        preds = []
        for csv in sorted((OUT_DIR / scenario.key / scenario.test_name / "predictions").glob(f"{model}_SP{start_point}_exp*.csv")):
            if csv.name.endswith(".metric.csv"):
                continue
            df = pd.read_csv(csv)
            if real is None:
                real = df["Real_Capacity"].to_numpy()
                cycles = df["Cycle"].to_numpy()
            preds.append(df["Predicted_Capacity"].to_numpy())
        if preds:
            min_len = min(len(p) for p in preds)
            model_frames[model] = np.vstack([p[:min_len] for p in preds]).mean(axis=0)

    if real is None or cycles is None:
        raise FileNotFoundError(f"No predictions found for {scenario.key} SP{start_point}")
    min_len = min([len(real), len(cycles)] + [len(v) for v in model_frames.values()])
    real = real[:min_len]
    cycles = cycles[:min_len]
    for model in list(model_frames):
        model_frames[model] = model_frames[model][:min_len]

    fig = plt.figure(figsize=(7.0, 6.4))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.55, 1.25], hspace=0.10)
    ax = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(cycles, real, color=real_color, linewidth=1.8, label="Real data", zorder=2)
    for model, y in model_frames.items():
        color, marker, ms = style[model]
        ax.plot(cycles, y, color=color, marker=marker, markersize=ms, markevery=max(1, len(cycles) // 25),
                linewidth=1.15, label=("NBD-Net(ours)" if model == "NBD-Net" else model), zorder=3, alpha=0.95)

    ax.axhline(y=threshold, color=threshold_color, linestyle="--", linewidth=1.05, label="Failure threshold", zorder=1)
    ax.axvline(x=start_point, color=sp_color, linestyle="-", linewidth=1.45, zorder=1)
    ax.set_ylabel("Capacity (Ah)", fontsize=14)
    ax.set_xlim(max(0, int(cycles.min()) - 60), int(cycles.max()) + 20)
    y_pad = max((real.max() - real.min()) * 0.22, 0.08)
    ax.set_ylim(max(0, min(real.min(), threshold) - y_pad), max(real.max(), *(v.max() for v in model_frames.values())) + y_pad)
    ax.tick_params(labelsize=12, direction="in", top=True, right=True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.55, color=grid_color, alpha=0.65)
    plt.setp(ax.get_xticklabels(), visible=False)

    handles, labels = ax.get_legend_handles_labels()
    legend_order = ["Failure threshold", "Real data", "GRU", "LSTM", "TCN", "CNN-LSTM", "Informer", "Transformer", "PatchTST", "NBD-Net(ours)"]
    idx = [labels.index(label) for label in legend_order if label in labels]
    ax.legend([handles[i] for i in idx], [labels[i] for i in idx], loc="upper right", ncol=3,
              fontsize=8.2, frameon=True, fancybox=False, edgecolor="#bfbfbf",
              facecolor="white", framealpha=0.92, handletextpad=0.35, columnspacing=0.75,
              labelspacing=0.28, borderpad=0.35)

    below = np.flatnonzero(real <= threshold)
    if len(below):
        center_idx = int(below[0])
    else:
        center_idx = int(np.argmin(np.abs(real - threshold)))
    half_window = max(12, min(40, len(cycles) // 8))
    lo_idx = max(0, center_idx - half_window)
    hi_idx = min(len(cycles) - 1, center_idx + half_window)
    if hi_idx <= lo_idx:
        hi_idx = min(len(cycles) - 1, lo_idx + 1)
    zoom_start = cycles[lo_idx]
    zoom_end = cycles[hi_idx]
    zmask = (cycles >= zoom_start) & (cycles <= zoom_end)
    vals = [real[zmask]] + [v[zmask] for v in model_frames.values()]
    zlo, zhi = np.nanpercentile(np.concatenate(vals), [4, 96])
    zlo = min(zlo, threshold)
    zhi = max(zhi, threshold)
    zpad = max((zhi - zlo) * 0.18, 0.015)
    axins = ax.inset_axes([0.14, 0.13, 0.46, 0.38])
    axins.plot(cycles, real, color=real_color, linewidth=1.55)
    for model, y in model_frames.items():
        color, marker, ms = style[model]
        axins.plot(cycles, y, color=color, marker=marker, markersize=max(ms - 0.4, 2.8),
                   markevery=max(1, len(cycles) // 80), linewidth=1.0, alpha=0.95)
    axins.axhline(y=threshold, color=threshold_color, linestyle="--", linewidth=0.95)
    axins.set_xlim(float(zoom_start), float(zoom_end))
    axins.set_ylim(float(zlo - zpad), float(zhi + zpad))
    axins.tick_params(labelsize=8.8, direction="in", top=True, right=True)
    axins.grid(True, linestyle="--", linewidth=0.45, color=grid_color, alpha=0.60)
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.75, linewidth=0.8)

    ax.text(0.965, 0.055, scenario.panel_label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right")

    all_errors = []
    for model, y in model_frames.items():
        color, marker, ms = style[model]
        error = y - real
        all_errors.append(error)
        ax_err.plot(cycles, error, color=color, linewidth=0.95, alpha=0.82, zorder=2)
        ax_err.scatter(cycles, error, color=color, marker=marker, s=(ms + 1.2) ** 2,
                       alpha=0.72, edgecolors="none", zorder=3)

    ax_err.axhline(y=0, color="#262626", linestyle="--", linewidth=0.95, zorder=1)
    ax_err.axvline(x=start_point, color=sp_color, linestyle="-", linewidth=1.2, alpha=0.85, zorder=1)
    ax_err.set_xlabel("Cycle", fontsize=14)
    ax_err.set_ylabel("Error (Ah)", fontsize=13)
    ax_err.tick_params(labelsize=11.5, direction="in", top=True, right=True)
    ax_err.grid(True, which="major", linestyle="--", linewidth=0.50, color=grid_color, alpha=0.65)
    lo, hi = np.nanpercentile(np.concatenate(all_errors), [1, 99])
    pad = max((hi - lo) * 0.18, 1e-4)
    ax_err.set_ylim(float(lo - pad), float(hi + pad))

    fig.align_ylabels([ax, ax_err])
    fig.tight_layout()
    out = FIG_DIR / f"{scenario.key}_{scenario.test_name}_SP{start_point}_capacity_error.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the user-specified baseline scripts plus NBD-Net under the same protocol.")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--representative-only", action="store_true",
                        help="Run only the representative start point used for the paper comparison figure.")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_rows = []
    for key in args.scenarios:
        scenario = SCENARIOS[key]
        module = importlib.import_module(scenario.module_name)
        scenario_args = get_args(scenario, args)
        scenario_args.patience = args.patience
        print(f"\n=== {scenario.dataset} {scenario.test_name} ({scenario.module_name}) ===")
        battery_data = load_battery_data(module, scenario)

        start_points = (scenario.default_plot_sp,) if args.representative_only else scenario.start_points
        for model in args.models:
            for sp in start_points:
                for exp in range(1, args.count + 1):
                    if args.plot_only:
                        continue
                    print(f"[{scenario.key}|{scenario.test_name}|{model}|SP{sp}|run {exp}/{args.count}]")
                    row = run_one(model, module, scenario, scenario_args, battery_data, sp, exp, device, force=args.force)
                    print(
                        f"  RMSE={row['RMSE']:.5f} MAE={row['MAE']:.5f} "
                        f"R2={row['R2']:.5f} AE={row['AE']} RE={row['RE']:.5f}"
                    )
                    raw_rows.append(row)

        metric_files = sorted((OUT_DIR / scenario.key / scenario.test_name / "predictions").glob("*.metric.csv"))
        if metric_files:
            raw = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
            raw.to_csv(OUT_DIR / f"{scenario.key}_raw.csv", index=False)
            sp_summary, overall_summary = summarize(raw)
            sp_summary.to_csv(OUT_DIR / f"{scenario.key}_summary_by_SP.csv", index=False)
            overall_summary.to_csv(OUT_DIR / f"{scenario.key}_summary_overall.csv", index=False)
            plot_points = start_points if not args.representative_only else (scenario.default_plot_sp or scenario.start_points[0],)
            for plot_sp in plot_points:
                plot_path = plot_capacity_error(scenario, plot_sp, args.models)
                print(f"  Figure: {plot_path}")

    all_metric_files = sorted(p for p in OUT_DIR.glob("*_raw.csv") if p.name != "specified_protocol_raw.csv")
    if all_metric_files:
        all_raw = pd.concat([pd.read_csv(p) for p in all_metric_files], ignore_index=True)
        all_raw.to_csv(OUT_DIR / "specified_protocol_raw.csv", index=False)
        sp_summary, overall_summary = summarize(all_raw)
        sp_summary.to_csv(OUT_DIR / "specified_protocol_summary_by_SP.csv", index=False)
        overall_summary.to_csv(OUT_DIR / "specified_protocol_summary_overall.csv", index=False)
        print("\nOverall summary:")
        print(overall_summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
