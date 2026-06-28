#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_MODULES = ["pandas", "numpy", "tensorflow", "sklearn"]
missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing model dependencies: " + ", ".join(missing))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "transactions.csv"
PREV2_PATH = PROJECT_DIR / "outputs" / "e07_prev2_features.csv"
EXACT_PREV_PATH = PROJECT_DIR / "outputs" / "e09_exact_prev_features.csv"
E10_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_e10_outlier_signal_experiments.py"

ARTIFACT_DIR = PROJECT_DIR / "models" / "f18_canonical_huber_010"
MODEL_PATH = ARTIFACT_DIR / "keras_model.keras"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
FEATURE_SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
NUMERIC_MEDIANS_PATH = ARTIFACT_DIR / "numeric_medians.json"
LOOKUP_VOCAB_DIR = ARTIFACT_DIR / "lookup_vocabulary"
METRICS_PATH = ARTIFACT_DIR / "eval_metrics.csv"
SAMPLE_INPUT_PATH = ARTIFACT_DIR / "sample_input.json"
SMOKE_PREDICTIONS_PATH = ARTIFACT_DIR / "smoke_predictions.csv"

MODEL_VERSION = "canonical_F18_reference_huber_010"
SOURCE_EXPERIMENT = "F18_reference_huber_010"
RANDOM_STATE = 42
SEED_OFFSET = 183
BATCH_SIZE = int(os.environ.get("F18_ARTIFACT_BATCH_SIZE", "8192"))
MAX_EPOCHS = int(os.environ.get("F18_ARTIFACT_MAX_EPOCHS", "30"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("F18_ARTIFACT_EARLY_STOPPING_PATIENCE", "4"))
TRAIN_VERBOSE = int(os.environ.get("F18_ARTIFACT_TRAIN_VERBOSE", "2"))
SMOKE_LIMIT = int(os.environ.get("F18_ARTIFACT_SMOKE_LIMIT", "0"))
ERROR_RATE_THRESHOLDS = [0.10, 0.20, 0.30, 0.50]
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)


def load_e10_module():
    spec = importlib.util.spec_from_file_location("e10_runner_for_artifact", E10_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E10_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e10 = load_e10_module()
e09 = e10.e09

CONFIG = {
    "experiment_name": SOURCE_EXPERIMENT,
    "numeric_features": list(e10.F18_FEATURES),
    "base_log_feature": "log_complex_prev_price_per_m2",
    "embedding_features": e09.BASE_EMBEDDING_FEATURES,
    "embedding_dims": e09.EMBEDDING_DIMS,
    "learning_rate": 0.001,
    "dense_units": [128, 64],
    "seed_offset": SEED_OFFSET,
    "loss": "huber_010",
}


def split_frames(policy_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {
        "train": policy_df.loc[policy_df["deal_date"] <= "2023-12-31"],
        "valid": policy_df.loc[(policy_df["deal_date"] >= "2024-01-01") & (policy_df["deal_date"] <= "2024-12-31")],
        "test": policy_df.loc[(policy_df["deal_date"] >= "2025-01-01") & (policy_df["deal_date"] <= "2025-12-31")],
        "recent_holdout": policy_df.loc[policy_df["deal_date"] >= "2026-01-01"],
    }
    for name, frame in splits.items():
        if frame.empty:
            raise RuntimeError(f"empty split: {name}")
    if SMOKE_LIMIT > 0:
        out = {}
        for name, frame in splits.items():
            out[name] = frame.sample(n=min(SMOKE_LIMIT, len(frame)), random_state=RANDOM_STATE).sort_values("deal_date").copy()
        return out
    return {key: value.copy() for key, value in splits.items()}


def load_training_frame() -> pd.DataFrame:
    base_usecols = [
        "transaction_id",
        "complex_id",
        "legal_dong_code",
        "sgg_code",
        "area_m2",
        "floor",
        "age_years",
        "deal_date",
        "deal_ym",
        "trade_type",
        "is_cancelled",
        "price_total",
        "price_per_m2",
        "complex_prev_price_per_m2",
        "complex_prev_missing",
        "prev_deal_gap_days",
    ]
    base_dtypes = {
        "transaction_id": "string",
        "complex_id": "string",
        "legal_dong_code": "string",
        "sgg_code": "string",
        "area_m2": "float32",
        "floor": "float32",
        "age_years": "float32",
        "trade_type": "string",
        "is_cancelled": "Int8",
        "price_total": "float32",
        "price_per_m2": "float32",
        "complex_prev_price_per_m2": "float32",
        "complex_prev_missing": "Int8",
        "prev_deal_gap_days": "float32",
    }
    raw_df = pd.read_csv(DATA_PATH, usecols=base_usecols, dtype=base_dtypes, parse_dates=["deal_date"])
    prev2_df = pd.read_csv(PREV2_PATH, dtype={"transaction_id": "string", "prev2_missing": "Int8"}, parse_dates=["prev2_source_deal_date"])
    exact_df = pd.read_csv(
        EXACT_PREV_PATH,
        dtype={"transaction_id": "string", "exact_prev1_missing": "Int8", "exact_prev2_missing": "Int8"},
        parse_dates=["exact_prev1_source_deal_date", "exact_prev2_source_deal_date"],
    )
    if not raw_df["transaction_id"].is_unique:
        raise RuntimeError("transactions transaction_id is not unique")
    if not prev2_df["transaction_id"].is_unique:
        raise RuntimeError("prev2 transaction_id is not unique")
    if not exact_df["transaction_id"].is_unique:
        raise RuntimeError("exact prev transaction_id is not unique")
    if len(exact_df) != len(raw_df):
        raise RuntimeError(f"exact sidecar row mismatch: exact={len(exact_df)} raw={len(raw_df)}")

    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    if int(model_df["prev2_missing"].isna().sum()) != 0:
        raise RuntimeError("prev2 sidecar join produced missing rows")
    if int(model_df["exact_prev1_missing"].isna().sum()) != 0:
        raise RuntimeError("exact sidecar join produced missing rows")
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    return e09.add_model_features(model_df)


def metric_row(split_df: pd.DataFrame, pred_log: np.ndarray, split_name: str) -> dict[str, Any]:
    y_true = split_df["target"].to_numpy(dtype="float64")
    pred_log = np.asarray(pred_log, dtype="float64").reshape(-1)
    pred_ppm = np.exp(pred_log)
    actual_ppm = split_df["price_per_m2"].to_numpy(dtype="float64")
    pred_total = pred_ppm * split_df["area_m2"].to_numpy(dtype="float64")
    actual_total = split_df["price_total"].to_numpy(dtype="float64")
    abs_pct = np.abs((pred_ppm - actual_ppm) / actual_ppm)
    out = {
        "model_version": MODEL_VERSION,
        "split": split_name,
        "rows": len(split_df),
        "log_mae": float(mean_absolute_error(y_true, pred_log)),
        "log_rmse": float(math.sqrt(mean_squared_error(y_true, pred_log))),
        "price_per_m2_mae": float(mean_absolute_error(actual_ppm, pred_ppm)),
        "total_price_mae_manwon": float(mean_absolute_error(actual_total, pred_total)),
        "abs_pct_error_p95": float(np.quantile(abs_pct, 0.95)),
        "abs_pct_error_p99": float(np.quantile(abs_pct, 0.99)),
    }
    for threshold in ERROR_RATE_THRESHOLDS:
        out[f"error_gt_{int(threshold * 100)}pct_rate"] = float((abs_pct > threshold).mean())
    return out


def save_lookup_vocabulary(lookups: dict[str, keras.layers.StringLookup]) -> dict[str, list[str]]:
    LOOKUP_VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    vocabularies = {}
    for feature, lookup in lookups.items():
        vocab = [str(value) for value in lookup.get_vocabulary()]
        vocabularies[feature] = vocab
        (LOOKUP_VOCAB_DIR / f"{feature}.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return vocabularies


def save_sample_input(splits: dict[str, pd.DataFrame], medians: pd.Series) -> None:
    sample = splits["recent_holdout"].iloc[0]
    numeric_values = {
        feature: None if pd.isna(sample[feature]) else float(sample[feature])
        for feature in e09.numeric_features(CONFIG)
    }
    embedding_values = {
        feature: str(sample[feature]) if pd.notna(sample[feature]) else "missing"
        for feature in e09.embedding_features(CONFIG)
    }
    payload = {
        "model_version": MODEL_VERSION,
        "base_log_feature": CONFIG["base_log_feature"],
        "base_log_value": float(sample[CONFIG["base_log_feature"]]) if pd.notna(sample[CONFIG["base_log_feature"]]) else float(medians[CONFIG["base_log_feature"]]),
        "numeric_features": numeric_values,
        "embedding_features": embedding_values,
        "area_m2": float(sample["area_m2"]),
        "actual_price_per_m2": float(sample["price_per_m2"]),
        "actual_price_total_manwon": float(sample["price_total"]),
        "transaction_id": str(sample["transaction_id"]),
    }
    SAMPLE_INPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    print("python", sys.version)
    print("tensorflow", tf.__version__)
    print("pandas", pd.__version__)
    print("project", PROJECT_DIR)
    print("artifact_dir", ARTIFACT_DIR)
    print("max_epochs", MAX_EPOCHS, "batch_size", BATCH_SIZE, "smoke_limit", SMOKE_LIMIT)

    if ARTIFACT_DIR.exists() and os.environ.get("F18_ARTIFACT_FORCE", "0") != "1":
        raise SystemExit(f"{ARTIFACT_DIR} exists. Set F18_ARTIFACT_FORCE=1 to overwrite.")
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    model_df = load_training_frame()
    splits = split_frames(model_df)
    split_counts = {name: len(frame) for name, frame in splits.items()}
    print("split_counts", split_counts)

    medians = e09.numeric_medians_for(CONFIG, splits)
    train_inputs, normalizer, lookups = e09.build_preprocessors(CONFIG, splits["train"], medians)
    model = e10.build_model(CONFIG, normalizer, lookups)
    valid_inputs = e09.make_inputs(splits["valid"], CONFIG, medians)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    history = model.fit(
        train_inputs,
        e09.y_for(splits["train"], medians, CONFIG),
        validation_data=(valid_inputs, e09.y_for(splits["valid"], medians, CONFIG)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=TRAIN_VERBOSE,
    )
    model.save(MODEL_PATH)
    vocabularies = save_lookup_vocabulary(lookups)

    metrics = []
    smoke_rows = []
    for split_name in SPLIT_ORDER:
        inputs = e09.make_inputs(splits[split_name], CONFIG, medians)
        raw_pred = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        pred_log = e09.final_log_pred(splits[split_name], raw_pred, medians, CONFIG)
        if not np.isfinite(pred_log).all():
            raise RuntimeError(f"non-finite predictions for {split_name}")
        metrics.append(metric_row(splits[split_name], pred_log, split_name))
        sample_size = min(10, len(splits[split_name]))
        for i in range(sample_size):
            smoke_rows.append(
                {
                    "split": split_name,
                    "transaction_id": str(splits[split_name].iloc[i]["transaction_id"]),
                    "actual_price_per_m2": float(splits[split_name].iloc[i]["price_per_m2"]),
                    "pred_price_per_m2": float(np.exp(pred_log[i])),
                    "abs_pct_error": float(abs(np.exp(pred_log[i]) - splits[split_name].iloc[i]["price_per_m2"]) / splits[split_name].iloc[i]["price_per_m2"]),
                }
            )
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)
    pd.DataFrame(smoke_rows).to_csv(SMOKE_PREDICTIONS_PATH, index=False)

    FEATURE_SCHEMA_PATH.write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "source_experiment": SOURCE_EXPERIMENT,
                "target": "log(price_per_m2)",
                "prediction_output": "residual log(price_per_m2) added to base_log_feature",
                "base_log_feature": CONFIG["base_log_feature"],
                "numeric_features": e09.numeric_features(CONFIG),
                "embedding_features": e09.embedding_features(CONFIG),
                "embedding_dims": CONFIG["embedding_dims"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    NUMERIC_MEDIANS_PATH.write_text(json.dumps({k: float(v) for k, v in medians.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_sample_input(splits, medians)

    metadata = {
        "model_version": MODEL_VERSION,
        "source_experiment": SOURCE_EXPERIMENT,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "project_dir": str(PROJECT_DIR),
        "model_path": str(MODEL_PATH),
        "max_epochs": MAX_EPOCHS,
        "epochs_ran": len(history.history.get("loss", [])),
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "Adam",
        "learning_rate": CONFIG["learning_rate"],
        "loss": "Huber(delta=0.10)",
        "random_seed": RANDOM_STATE + SEED_OFFSET,
        "split": "train<=2023, valid=2024, test=2025, recent_holdout>=2026",
        "policy": "is_cancelled == 0 and trade_type in [중개거래, unknown]",
        "split_counts": split_counts,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "lookup_vocabulary_sizes": {feature: len(vocab) for feature, vocab in vocabularies.items()},
        "metrics": metrics_df.to_dict(orient="records"),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "_SUCCESS").write_text("trained\n", encoding="utf-8")

    print("saved_model", MODEL_PATH)
    print(metrics_df[["split", "rows", "log_mae", "abs_pct_error_p95", "abs_pct_error_p99", "error_gt_10pct_rate", "error_gt_20pct_rate"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
