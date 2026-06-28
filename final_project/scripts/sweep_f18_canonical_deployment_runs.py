#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

import train_f18_canonical_model_artifact as artifact


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "models" / "f18_canonical_huber_010_runs"
BEST_DIR = PROJECT_DIR / "models" / "f18_canonical_huber_010_best_attempt"
SUMMARY_CSV = RUNS_DIR / "sweep_summary.csv"
SUMMARY_MD = RUNS_DIR / "sweep_summary.md"
HISTORICAL_RECENT = {
    "log_mae": 0.06177453101105649,
    "p95": 0.18807662850545506,
    "p99": 0.34558166944175545,
    "gt10": 0.17747340882616916,
    "gt20": 0.04295166803521302,
}

DEFAULT_RUNS = [
    {"run_name": "huber010_seed183_direct", "seed_offset": 183, "loss": "huber_010"},
    {"run_name": "huber010_seed184_direct", "seed_offset": 184, "loss": "huber_010"},
    {"run_name": "huber010_seed185_direct", "seed_offset": 185, "loss": "huber_010"},
    {"run_name": "huber010_seed186_direct", "seed_offset": 186, "loss": "huber_010"},
    {"run_name": "huber010_seed187_direct", "seed_offset": 187, "loss": "huber_010"},
    {"run_name": "huber005_seed182_direct", "seed_offset": 182, "loss": "huber_005"},
    {"run_name": "mse_seed183_direct", "seed_offset": 183, "loss": "mse"},
]

MAX_EPOCHS = int(os.environ.get("F18_SWEEP_MAX_EPOCHS", str(artifact.MAX_EPOCHS)))
BATCH_SIZE = int(os.environ.get("F18_SWEEP_BATCH_SIZE", str(artifact.BATCH_SIZE)))
PATIENCE = int(os.environ.get("F18_SWEEP_EARLY_STOPPING_PATIENCE", str(artifact.EARLY_STOPPING_PATIENCE)))
TRAIN_VERBOSE = int(os.environ.get("F18_SWEEP_TRAIN_VERBOSE", "2"))
FORCE = os.environ.get("F18_SWEEP_FORCE", "0") == "1"


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def run_specs() -> list[dict[str, Any]]:
    raw = os.environ.get("F18_SWEEP_RUNS", "").strip()
    if not raw:
        return DEFAULT_RUNS
    specs = []
    for item in raw.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise SystemExit("F18_SWEEP_RUNS format: run_name:loss:seed_offset,...")
        specs.append({"run_name": parts[0], "loss": parts[1], "seed_offset": int(parts[2])})
    return specs


def build_model(config: dict[str, Any], normalizer, lookups: dict[str, keras.layers.StringLookup]):
    tf.keras.utils.set_random_seed(artifact.RANDOM_STATE + int(config.get("seed_offset", 0)))
    numeric_input = keras.Input(shape=(len(artifact.e09.numeric_features(config)),), name="numeric_input", dtype="float32")
    parts = [normalizer(numeric_input)]
    inputs = [numeric_input]
    for feature in artifact.e09.embedding_features(config):
        inp = keras.Input(shape=(1,), name=f"{feature}_input", dtype=tf.string)
        idx = lookups[feature](inp)
        dim = int(config["embedding_dims"].get(feature, artifact.e09.EMBEDDING_DIMS[feature]))
        emb = keras.layers.Embedding(lookups[feature].vocabulary_size(), dim, name=f"{feature}_embedding")(idx)
        inputs.append(inp)
        parts.append(keras.layers.Flatten(name=f"{feature}_flatten")(emb))
    x = keras.layers.Concatenate(name="feature_concat")(parts)
    for unit in config["dense_units"]:
        x = keras.layers.Dense(unit, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = keras.layers.Dropout(0.10 if unit >= 128 else 0.05)(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    loss_name = config.get("loss", "huber_010")
    if loss_name == "huber_005":
        loss: Any = keras.losses.Huber(delta=0.05)
    elif loss_name == "huber_010":
        loss = keras.losses.Huber(delta=0.10)
    elif loss_name == "mse":
        loss = "mse"
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss=loss, metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


def metric_row(model_version: str, split_df: pd.DataFrame, pred_log: np.ndarray, split_name: str) -> dict[str, Any]:
    y_true = split_df["target"].to_numpy(dtype="float64")
    pred_log = np.asarray(pred_log, dtype="float64").reshape(-1)
    pred_ppm = np.exp(pred_log)
    actual_ppm = split_df["price_per_m2"].to_numpy(dtype="float64")
    pred_total = pred_ppm * split_df["area_m2"].to_numpy(dtype="float64")
    actual_total = split_df["price_total"].to_numpy(dtype="float64")
    abs_pct = np.abs((pred_ppm - actual_ppm) / actual_ppm)
    return {
        "model_version": model_version,
        "split": split_name,
        "rows": len(split_df),
        "log_mae": float(mean_absolute_error(y_true, pred_log)),
        "log_rmse": float(math.sqrt(mean_squared_error(y_true, pred_log))),
        "price_per_m2_mae": float(mean_absolute_error(actual_ppm, pred_ppm)),
        "total_price_mae_manwon": float(mean_absolute_error(actual_total, pred_total)),
        "abs_pct_error_p95": float(np.quantile(abs_pct, 0.95)),
        "abs_pct_error_p99": float(np.quantile(abs_pct, 0.99)),
        "error_gt_10pct_rate": float((abs_pct > 0.10).mean()),
        "error_gt_20pct_rate": float((abs_pct > 0.20).mean()),
        "error_gt_30pct_rate": float((abs_pct > 0.30).mean()),
        "error_gt_50pct_rate": float((abs_pct > 0.50).mean()),
    }


def save_common_artifacts(run_dir: Path, config: dict[str, Any], medians: pd.Series, splits: dict[str, pd.DataFrame], lookups: dict[str, keras.layers.StringLookup], metrics_df: pd.DataFrame, history: keras.callbacks.History, elapsed: float) -> None:
    vocab_dir = run_dir / "lookup_vocabulary"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_sizes = {}
    for feature, lookup in lookups.items():
        vocab = [str(value) for value in lookup.get_vocabulary()]
        vocab_sizes[feature] = len(vocab)
        (vocab_dir / f"{feature}.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    (run_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "model_version": config["model_version"],
                "source_experiment": config["experiment_name"],
                "target": "log(price_per_m2)",
                "prediction_output": "residual log(price_per_m2) added to base_log_feature",
                "base_log_feature": config["base_log_feature"],
                "numeric_features": artifact.e09.numeric_features(config),
                "embedding_features": artifact.e09.embedding_features(config),
                "embedding_dims": config["embedding_dims"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "numeric_medians.json").write_text(json.dumps({k: float(v) for k, v in medians.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact.SAMPLE_INPUT_PATH = run_dir / "sample_input.json"
    artifact.CONFIG = config
    artifact.save_sample_input(splits, medians)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_version": config["model_version"],
                "source_experiment": config["experiment_name"],
                "run_name": config["run_name"],
                "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "max_epochs": MAX_EPOCHS,
                "epochs_ran": len(history.history.get("loss", [])),
                "batch_size": BATCH_SIZE,
                "early_stopping_patience": PATIENCE,
                "optimizer": "Adam",
                "learning_rate": config["learning_rate"],
                "loss": config["loss"],
                "random_seed": artifact.RANDOM_STATE + int(config["seed_offset"]),
                "split": "train<=2023, valid=2024, test=2025, recent_holdout>=2026",
                "policy": "is_cancelled == 0 and trade_type in [중개거래, unknown]",
                "split_counts": {name: len(frame) for name, frame in splits.items()},
                "history": {k: [float(x) for x in v] for k, v in history.history.items()},
                "lookup_vocabulary_sizes": vocab_sizes,
                "metrics": metrics_df.to_dict(orient="records"),
                "elapsed_seconds": round(elapsed, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "_SUCCESS").write_text("trained\n", encoding="utf-8")


def train_one(config: dict[str, Any], splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tf.keras.backend.clear_session()
    run_dir = RUNS_DIR / config["run_name"]
    if run_dir.exists() and FORCE:
        shutil.rmtree(run_dir)
    if (run_dir / "_SUCCESS").exists() and not FORCE:
        print("skip_existing", run_dir)
        return pd.read_csv(run_dir / "eval_metrics.csv")
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("\n===", config["run_name"], "loss", config["loss"], "seed", config["seed_offset"], "===")
    medians = artifact.e09.numeric_medians_for(config, splits)
    train_inputs, normalizer, lookups = artifact.e09.build_preprocessors(config, splits["train"], medians)
    model = build_model(config, normalizer, lookups)
    valid_inputs = artifact.e09.make_inputs(splits["valid"], config, medians)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    history = model.fit(
        train_inputs,
        artifact.e09.y_for(splits["train"], medians, config),
        validation_data=(valid_inputs, artifact.e09.y_for(splits["valid"], medians, config)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=TRAIN_VERBOSE,
    )
    model.save(run_dir / "keras_model.keras")
    metrics = []
    for split_name in artifact.SPLIT_ORDER:
        inputs = artifact.e09.make_inputs(splits[split_name], config, medians)
        raw_pred = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        pred_log = artifact.e09.final_log_pred(splits[split_name], raw_pred, medians, config)
        metrics.append(metric_row(config["model_version"], splits[split_name], pred_log, split_name))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(run_dir / "eval_metrics.csv", index=False)
    save_common_artifacts(run_dir, config, medians, splits, lookups, metrics_df, history, time.perf_counter() - start)
    recent = metrics_df.loc[metrics_df["split"].eq("recent_holdout")].iloc[0]
    print(
        "recent",
        config["run_name"],
        f"mae={recent['log_mae']:.6f}",
        f"p95={pct(recent['abs_pct_error_p95'])}",
        f"p99={pct(recent['abs_pct_error_p99'])}",
        f"gt10={pct(recent['error_gt_10pct_rate'])}",
        f"gt20={pct(recent['error_gt_20pct_rate'])}",
    )
    return metrics_df


def write_summary(all_metrics: pd.DataFrame) -> None:
    recent = all_metrics.loc[all_metrics["split"].eq("recent_holdout")].copy()
    recent["delta_mae_vs_historical"] = recent["log_mae"] - HISTORICAL_RECENT["log_mae"]
    recent["delta_p99_vs_historical"] = recent["abs_pct_error_p99"] - HISTORICAL_RECENT["p99"]
    recent["delta_gt20_vs_historical"] = recent["error_gt_20pct_rate"] - HISTORICAL_RECENT["gt20"]
    recent = recent.sort_values(["log_mae", "error_gt_20pct_rate", "abs_pct_error_p99"]).reset_index(drop=True)
    recent.to_csv(SUMMARY_CSV, index=False)

    best = recent.iloc[0]
    best_run = str(best["model_version"]).replace("canonical_F18_reference_huber_010__", "")
    best_dir = RUNS_DIR / best_run
    if BEST_DIR.exists():
        shutil.rmtree(BEST_DIR)
    shutil.copytree(best_dir, BEST_DIR)
    lines = [
        "# F18 deployment retrain sweep",
        "",
        "## Historical target",
        "",
        "| MAE(log) | p95 | p99 | >10% | >20% |",
        "| ---: | ---: | ---: | ---: | ---: |",
        f"| {HISTORICAL_RECENT['log_mae']:.6f} | {pct(HISTORICAL_RECENT['p95'])} | {pct(HISTORICAL_RECENT['p99'])} | {pct(HISTORICAL_RECENT['gt10'])} | {pct(HISTORICAL_RECENT['gt20'])} |",
        "",
        "## Best saved attempt",
        "",
        f"- run: `{best_run}`",
        f"- artifact: `{BEST_DIR}`",
        f"- recent_holdout MAE(log): `{best['log_mae']:.6f}`",
        f"- delta MAE vs historical: `{best['delta_mae_vs_historical']:.6f}`",
        "",
        "## Recent holdout ranking",
        "",
        "| rank | run | MAE(log) | p95 | p99 | >10% | >20% | d_MAE_hist |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in recent.iterrows():
        run = str(row["model_version"]).replace("canonical_F18_reference_huber_010__", "")
        lines.append(
            f"| {idx + 1} | {run} | {row['log_mae']:.6f} | {pct(row['abs_pct_error_p95'])} | {pct(row['abs_pct_error_p99'])} | "
            f"{pct(row['error_gt_10pct_rate'])} | {pct(row['error_gt_20pct_rate'])} | {row['delta_mae_vs_historical']:.6f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if RUNS_DIR.exists() and FORCE:
        shutil.rmtree(RUNS_DIR)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print("load_training_frame")
    model_df = artifact.load_training_frame()
    splits = artifact.split_frames(model_df)
    print("split_counts", {name: len(frame) for name, frame in splits.items()})

    all_frames = []
    for spec in run_specs():
        config = dict(artifact.CONFIG)
        config.update(spec)
        config["experiment_name"] = "F18_reference_huber_010"
        config["model_version"] = f"canonical_F18_reference_huber_010__{spec['run_name']}"
        all_frames.append(train_one(config, splits))
    all_metrics = pd.concat(all_frames, ignore_index=True)
    all_metrics.to_csv(RUNS_DIR / "all_eval_metrics.csv", index=False)
    write_summary(all_metrics)
    print("summary", SUMMARY_MD)
    print("best", BEST_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
