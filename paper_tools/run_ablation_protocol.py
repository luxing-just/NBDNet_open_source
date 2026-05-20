from __future__ import annotations

import argparse
import importlib.util
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
OUT_DIR = ROOT / "paper_experiment_results" / "ablation_protocol"
FIG_DIR = ROOT / "paper_figures_real" / "ablation_protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = ROOT / "RUL_NBDNet_ablation.py"


@dataclass(frozen=True)
class AblationScenario:
    battery: str
    start_points: tuple[int, ...]


SCENARIOS = [
    AblationScenario("CS2_35", (300, 400, 500)),
    AblationScenario("CS2_37", (300, 400, 500)),
]

ABLATIONS = ["A0", "A1", "A2", "A3", "A4", "A5"]


def load_ablation_module():
    spec = importlib.util.spec_from_file_location("user_ablation_module", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def rul_error(y_test: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[int, int, int, float]:
    def find_rul(seq):
        for i in range(len(seq) - 1):
            if seq[i] <= threshold:
                return i
        return len(seq)

    rul_real = find_rul(y_test)
    rul_pred = find_rul(y_pred)
    ae = abs(rul_real - rul_pred)
    re = min(ae / rul_real, 1.0) if rul_real > 0 else 1.0
    return int(rul_real), int(rul_pred), int(ae), float(re)


def make_args(cli_args: argparse.Namespace, battery: str) -> SimpleNamespace:
    return SimpleNamespace(
        seed=1,
        seq_len=64,
        pred_len=1,
        d_model=64,
        n_local_layers=3,
        local_kernel=7,
        local_window=9,
        n_global_layers=2,
        n_trend_slots=2,
        n_transition_slots=2,
        n_fluct_slots=2,
        dropout=0.1,
        count=cli_args.count,
        batch_size=128,
        Rated_Capacity=1.1,
        test_name=battery,
        start_point_list=list(cli_args.start_points),
        max_epochs=cli_args.max_epochs,
        precision=cli_args.precision,
        patience=cli_args.patience,
    )


def prediction_path(battery: str, ablation: str, start_point: int, exp: int) -> Path:
    return OUT_DIR / battery / ablation / "predictions" / f"{ablation}_SP{start_point}_exp{exp}.csv"


def metric_path(battery: str, ablation: str, start_point: int, exp: int) -> Path:
    return prediction_path(battery, ablation, start_point, exp).with_suffix(".metric.csv")


def build_model(module, training_ds, args, ablation: str):
    return module.NBDNetAblationModel.from_dataset(
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
        ablation=ablation,
        learning_rate=0.001,
        loss=SMAPE(),
    )


def run_one(module, battery_data, battery: str, ablation: str, start_point: int, exp: int,
            cli_args: argparse.Namespace, device: torch.device, force: bool = False) -> dict:
    mpath = metric_path(battery, ablation, start_point, exp)
    if mpath.exists() and not force:
        return pd.read_csv(mpath).iloc[0].to_dict()

    args = make_args(cli_args, battery)
    set_seed(exp)
    df_train, df_test, df_all = module.battery_data_process(battery_data, battery, start_point, args)
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

    model = build_model(module, training_ds, args, ablation)
    params_k = model.size() / 1e3
    ckpt_dir = OUT_DIR / battery / ablation / "checkpoints" / f"SP{start_point}_exp{exp}"
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
        deterministic="warn",
        logger=False,
    )
    t0 = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    train_time = time.time() - t0

    best_model = build_model(module, training_ds, args, ablation)
    ckpt = torch.load(trainer.checkpoint_callback.best_model_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    for key in ["network.revin.mean", "network.revin.stdev"]:
        state.pop(key, None)
    best_model.load_state_dict(state, strict=False)
    best_model = best_model.to(device).eval()
    preds = best_model.predict(test_dl, batch_size=256).cpu().numpy().reshape(-1)

    act_df = df_all.loc[df_all["Cycle"] >= start_point, ["Cycle", "target"]]
    y_true = act_df["target"].to_numpy(float) * args.Rated_Capacity
    y_pred = preds * args.Rated_Capacity
    cycles = act_df["Cycle"].to_numpy()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred, cycles = y_true[:n], y_pred[:n], cycles[:n]

    threshold = args.Rated_Capacity * 0.7
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    rul_r, rul_p, ae, re = rul_error(y_true, y_pred, threshold)
    row = {
        "battery": battery,
        "ablation": ablation,
        "name": module.ABLATION_CONFIGS[ablation]["name"],
        "start_point": start_point,
        "exp": exp,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "RUL_real": rul_r,
        "RUL_pred": rul_p,
        "AE": ae,
        "RE": re,
        "n_params_k": round(params_k, 2),
        "train_time_s": round(train_time, 3),
    }

    ppath = prediction_path(battery, ablation, start_point, exp)
    ppath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Cycle": cycles,
        "Real_Capacity": y_true,
        "Predicted_Capacity": y_pred,
        "Error": y_pred - y_true,
    }).to_csv(ppath, index=False)
    pd.DataFrame([row]).to_csv(mpath, index=False)

    del model, best_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def summarize() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(OUT_DIR.glob("*/*/predictions/*.metric.csv"))
    raw = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    raw.to_csv(OUT_DIR / "ablation_raw.csv", index=False)
    metrics = ["RMSE", "MAE", "R2", "AE", "RE", "n_params_k", "train_time_s"]
    by_sp = raw.groupby(["battery", "start_point", "ablation", "name"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        n_exp=("exp", "count"),
    )
    overall = raw.groupby(["ablation", "name"], as_index=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        n_exp=("exp", "count"),
    )
    order = {tag: i for i, tag in enumerate(ABLATIONS)}
    by_sp["order"] = by_sp["ablation"].map(order)
    overall["order"] = overall["ablation"].map(order)
    by_sp = by_sp.sort_values(["battery", "start_point", "order"]).drop(columns="order")
    overall = overall.sort_values("order").drop(columns="order")
    by_sp.to_csv(OUT_DIR / "ablation_summary_by_SP.csv", index=False)
    overall.to_csv(OUT_DIR / "ablation_summary_overall.csv", index=False)
    return raw, overall


def plot_overall(overall: pd.DataFrame) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = "Times New Roman"
    if font not in {f.name for f in font_manager.fontManager.ttflist}:
        font = "serif"
    mpl.rcParams["font.family"] = font
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False

    labels = overall["ablation"].tolist()
    rmse = overall["RMSE_mean"].to_numpy()
    err = overall["RMSE_std"].to_numpy()
    colors = ["#1f77b4", "#8da0cb", "#fc8d62", "#66c2a5", "#e78ac3", "#a6d854", "#ffd92f"]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, rmse, yerr=err, capsize=4, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("RMSE (Ah)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Ablation study on CALCE CS2_35/CS2_37", fontsize=12)
    for xi, yi in zip(x, rmse):
        ax.text(xi, yi + err.max() * 0.08, f"{yi:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig9_ablation_rmse.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached ablation runner for the user-provided CALCE ablation code.")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--batteries", nargs="+", default=[scenario.battery for scenario in SCENARIOS])
    parser.add_argument("--ablations", nargs="+", default=ABLATIONS)
    parser.add_argument("--start-points", nargs="+", type=int, default=[300, 400, 500])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    module = load_ablation_module()
    battery_data = np.load(ROOT / "data/CALCE data/CALCE_Data.npy", allow_pickle=True).item()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.plot_only:
        selected_scenarios = [
            AblationScenario(scenario.battery, tuple(sp for sp in scenario.start_points if sp in args.start_points))
            for scenario in SCENARIOS
            if scenario.battery in args.batteries
        ]
        selected_scenarios = [scenario for scenario in selected_scenarios if scenario.start_points]
        selected_ablations = [ablation for ablation in ABLATIONS if ablation in args.ablations]
        total = sum(len(scenario.start_points) for scenario in selected_scenarios) * len(selected_ablations) * args.count
        done = len(list(OUT_DIR.glob("*/*/predictions/*.metric.csv"))) if not args.force else 0
        for scenario in selected_scenarios:
            for ablation in selected_ablations:
                for sp in scenario.start_points:
                    for exp in range(1, args.count + 1):
                        print(f"[{scenario.battery}|{ablation}|SP{sp}|run {exp}/{args.count}] {done}/{total}")
                        row = run_one(module, battery_data, scenario.battery, ablation, sp, exp, args, device, force=args.force)
                        done += 1
                        print(f"  RMSE={row['RMSE']:.5f} MAE={row['MAE']:.5f} R2={row['R2']:.5f} RE={row['RE']:.5f}")

    if not args.no_summary:
        raw, overall = summarize()
        fig = plot_overall(overall)
        print(f"raw rows={len(raw)} min_R2={raw['R2'].min():.6f} fig={fig}", flush=True)
        print(overall.round(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
