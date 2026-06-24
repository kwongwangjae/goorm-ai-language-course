#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REQUIRED_MODULES = ["pandas", "numpy", "tensorflow", "sklearn"]
missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing notebook/model dependencies: "
        + ", ".join(missing)
        + ". Run this in the same kernel that executed 07_test_prev2_transaction_features.ipynb."
    )

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras


current_dir = Path.cwd()
if current_dir.name == "final_project":
    PROJECT_DIR = current_dir
elif (current_dir / "final_project").exists():
    PROJECT_DIR = current_dir / "final_project"
else:
    PROJECT_DIR = Path("/Users/gwongwangjae/goorm-ai-language-course/final_project")

DATA_PATH = PROJECT_DIR / "data" / "processed" / "transactions.csv"
PREV2_PATH = PROJECT_DIR / "outputs" / "e07_prev2_features.csv"
EXACT_PREV_PATH = PROJECT_DIR / "outputs" / "e09_exact_prev_features.csv"
EXACT_PREV_BUILDER_PATH = PROJECT_DIR / "scripts" / "build_e09_exact_prev_features.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "e09_exact_prev_metrics.csv"
GROUP_METRICS_PATH = OUTPUT_DIR / "e09_exact_prev_group_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "e09_exact_prev_summary.md"

RUN_MODE = os.environ.get("E09_RUN_MODE", "full").strip().lower()
REBUILD_EXACT_PREV_FEATURES = os.environ.get("E09_REBUILD_EXACT_PREV_FEATURES", "0") == "1"
EXPERIMENT_FILTER = {
    name.strip()
    for name in os.environ.get("E09_EXPERIMENTS", "").split(",")
    if name.strip()
}
RANDOM_STATE = 42
SMOKE_LIMITS = {"train": 200_000, "valid": 50_000, "test": 50_000, "recent_holdout": 50_000}
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]
ERROR_RATE_THRESHOLDS = [0.10, 0.20, 0.30, 0.50]
BATCH_SIZE = int(os.environ.get("E09_BATCH_SIZE", "8192"))
MAX_EPOCHS = int(os.environ.get("E09_MAX_EPOCHS", "30"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("E09_EARLY_STOPPING_PATIENCE", "4"))
TRAIN_VERBOSE = int(os.environ.get("E09_TRAIN_VERBOSE", "2"))
REFERENCE_E07_F10_LOG_MAE = {"valid": 0.0590586680029334, "test": 0.0588961191284131, "recent_holdout": 0.0627029597139034}

assert RUN_MODE in {"smoke", "full"}, RUN_MODE
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)


def md_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float", "float32", "float64"]).columns:
        x[col] = x[col].map(lambda v: format(v, floatfmt) if pd.notna(v) else "")
    x = x.astype("string").fillna("")
    lines = ["| " + " | ".join(x.columns) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in x.values.tolist()]
    return "\n".join(lines)


def ensure_exact_prev_sidecar() -> None:
    if EXACT_PREV_PATH.exists() and not REBUILD_EXACT_PREV_FEATURES:
        return
    subprocess.run([sys.executable, str(EXACT_PREV_BUILDER_PATH)], cwd=PROJECT_DIR, check=True)


def gap_bucket(days: pd.Series) -> pd.Series:
    bucket = pd.Series("missing", index=days.index, dtype="string")
    bucket[(days >= 0) & (days <= 30)] = "0-30"
    bucket[(days >= 31) & (days <= 90)] = "31-90"
    bucket[(days >= 91) & (days <= 180)] = "91-180"
    bucket[(days >= 181) & (days <= 365)] = "181-365"
    bucket[days >= 366] = "366+"
    return bucket.fillna("missing").astype("string")


def gap_plus_bucket(days: pd.Series) -> pd.Series:
    bins = [-np.inf, 0, 30, 90, 180, 365, 730, np.inf]
    labels = ["missing_or_negative", "1-30", "31-90", "91-180", "181-365", "366-730", "731+"]
    return pd.cut(days.fillna(-1), bins=bins, labels=labels).astype("string").fillna("missing")


def add_model_features(input_df: pd.DataFrame) -> pd.DataFrame:
    out = input_df.copy()
    out["target"] = np.log(out["price_per_m2"].astype("float64"))
    out["is_basement_floor"] = (out["floor"] < 0).astype("float32")

    wide_prev1_price = out["complex_prev_price_per_m2"].astype("float64")
    wide_prev2_price = out["complex_prev2_price_per_m2"].astype("float64")
    exact_prev1_price = out["exact_prev1_price_per_m2"].astype("float64")
    exact_prev2_price = out["exact_prev2_price_per_m2"].astype("float64")

    out["log_complex_prev_price_per_m2"] = np.where(wide_prev1_price > 0, np.log(wide_prev1_price), np.nan)
    out["log_complex_prev2_price_per_m2"] = np.where(wide_prev2_price > 0, np.log(wide_prev2_price), np.nan)
    out["log_exact_prev1_price_per_m2"] = np.where(exact_prev1_price > 0, np.log(exact_prev1_price), np.nan)
    out["log_exact_prev2_price_per_m2"] = np.where(exact_prev2_price > 0, np.log(exact_prev2_price), np.nan)

    out["complex_prev_missing"] = out["complex_prev_missing"].fillna(1).astype("float32")
    out["prev2_missing"] = out["prev2_missing"].fillna(1).astype("float32")
    out["exact_prev1_missing"] = out["exact_prev1_missing"].fillna(1).astype("float32")
    out["exact_prev2_missing"] = out["exact_prev2_missing"].fillna(1).astype("float32")
    out["wide_prev1_present_exact_missing"] = out["wide_prev1_present_exact_missing"].fillna(0).astype("float32")
    out["exact_prev1_present_wide_missing"] = out["exact_prev1_present_wide_missing"].fillna(0).astype("float32")

    out["prev_deal_gap_months"] = out["prev_deal_gap_days"].astype("float64") / 30.4375
    out["prev2_gap_months"] = out["prev2_gap_days"].astype("float64") / 30.4375
    out["prev1_prev2_gap_months"] = out["prev1_prev2_gap_days"].astype("float64") / 30.4375
    out["exact_prev1_gap_months"] = out["exact_prev1_gap_days"].astype("float64") / 30.4375
    out["exact_prev2_gap_months"] = out["exact_prev2_gap_days"].astype("float64") / 30.4375
    out["exact_prev1_prev2_gap_months"] = out["exact_prev1_prev2_gap_days"].astype("float64") / 30.4375

    out["fallback_prev1_missing"] = ((out["exact_prev1_missing"] >= 1) & (out["complex_prev_missing"] >= 1)).astype("float32")
    out["fallback_prev2_missing"] = ((out["exact_prev2_missing"] >= 1) & (out["prev2_missing"] >= 1)).astype("float32")
    out["log_fallback_prev1_price_per_m2"] = out["log_exact_prev1_price_per_m2"].where(out["exact_prev1_missing"] < 1, out["log_complex_prev_price_per_m2"])
    out["log_fallback_prev2_price_per_m2"] = out["log_exact_prev2_price_per_m2"].where(out["exact_prev2_missing"] < 1, out["log_complex_prev2_price_per_m2"])
    out["fallback_prev1_gap_months"] = out["exact_prev1_gap_months"].where(out["exact_prev1_missing"] < 1, out["prev_deal_gap_months"])
    out["fallback_prev2_gap_months"] = out["exact_prev2_gap_months"].where(out["exact_prev2_missing"] < 1, out["prev2_gap_months"])
    out["fallback_prev1_prev2_log_return"] = out["log_fallback_prev1_price_per_m2"] - out["log_fallback_prev2_price_per_m2"]
    out["fallback_prev1_prev2_gap_months"] = out["fallback_prev2_gap_months"] - out["fallback_prev1_gap_months"]

    out["prev_deal_gap_bucket"] = gap_bucket(out["prev_deal_gap_days"].astype("float64"))
    out["exact_prev1_missing_group"] = out["exact_prev1_missing"].astype("Int8").astype("string")
    out["wide_prev1_present_exact_missing_group"] = out["wide_prev1_present_exact_missing"].astype("Int8").astype("string")
    out["exact_prev1_gap_bucket_plus"] = gap_plus_bucket(out["exact_prev1_gap_days"].astype("float64"))
    out["prev1_gap_bucket_plus"] = gap_plus_bucket(out["prev_deal_gap_days"].astype("float64"))
    out["prev2_gap_bucket_plus"] = gap_plus_bucket(out["prev2_gap_days"].astype("float64"))
    for feature in ["legal_dong_code", "sgg_code", "prev_deal_gap_bucket"]:
        out[feature] = out[feature].fillna("missing").astype("string")
    return out


def split_frames(policy_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {
        "train": policy_df.loc[policy_df["deal_date"] <= "2023-12-31"],
        "valid": policy_df.loc[(policy_df["deal_date"] >= "2024-01-01") & (policy_df["deal_date"] <= "2024-12-31")],
        "test": policy_df.loc[(policy_df["deal_date"] >= "2025-01-01") & (policy_df["deal_date"] <= "2025-12-31")],
        "recent_holdout": policy_df.loc[policy_df["deal_date"] >= "2026-01-01"],
    }
    for name, frame in splits.items():
        assert len(frame) > 0, name
    return splits


def apply_smoke_sampling(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if RUN_MODE != "smoke":
        return {key: value.copy() for key, value in splits.items()}
    out = {}
    for name, frame in splits.items():
        limit = SMOKE_LIMITS[name]
        out[name] = frame.sample(n=limit, random_state=RANDOM_STATE).sort_values("deal_date") if len(frame) > limit else frame.copy()
    return out


WIDE_F10_FEATURES = [
    "area_m2",
    "floor",
    "is_basement_floor",
    "age_years",
    "log_complex_prev_price_per_m2",
    "complex_prev_missing",
    "prev_deal_gap_months",
    "log_complex_prev2_price_per_m2",
    "prev2_missing",
    "prev2_gap_months",
    "prev1_prev2_log_return",
    "prev1_prev2_gap_months",
]
EXACT_REPLACE_FEATURES = [
    "area_m2",
    "floor",
    "is_basement_floor",
    "age_years",
    "log_exact_prev1_price_per_m2",
    "exact_prev1_missing",
    "exact_prev1_gap_months",
    "log_exact_prev2_price_per_m2",
    "exact_prev2_missing",
    "exact_prev2_gap_months",
    "exact_prev1_prev2_log_return",
    "exact_prev1_prev2_gap_months",
]
EXACT_ADDITIVE_FEATURES = WIDE_F10_FEATURES + [
    "log_exact_prev1_price_per_m2",
    "exact_prev1_missing",
    "exact_prev1_gap_months",
    "log_exact_prev2_price_per_m2",
    "exact_prev2_missing",
    "exact_prev2_gap_months",
    "exact_prev1_prev2_log_return",
    "exact_prev1_prev2_gap_months",
    "exact_prev1_area_abs_diff",
    "exact_prev2_area_abs_diff",
    "wide_prev1_present_exact_missing",
]
FALLBACK_FEATURES = [
    "area_m2",
    "floor",
    "is_basement_floor",
    "age_years",
    "log_fallback_prev1_price_per_m2",
    "fallback_prev1_missing",
    "fallback_prev1_gap_months",
    "log_fallback_prev2_price_per_m2",
    "fallback_prev2_missing",
    "fallback_prev2_gap_months",
    "fallback_prev1_prev2_log_return",
    "fallback_prev1_prev2_gap_months",
    "exact_prev1_missing",
    "exact_prev2_missing",
    "wide_prev1_present_exact_missing",
]
BASE_EMBEDDING_FEATURES = ["legal_dong_code", "sgg_code", "prev_deal_gap_bucket"]
EMBEDDING_DIMS = {"legal_dong_code": 16, "sgg_code": 8, "prev_deal_gap_bucket": 3}
HARD_LEAKAGE_COLUMNS = {
    "target",
    "price_total",
    "price_per_m2",
    "deal_date",
    "deal_ym",
    "transaction_id",
    "trade_type",
    "is_cancelled",
    "complex_id",
    "complex_prev_price_per_m2",
    "prev_deal_gap_days",
    "complex_prev2_price_per_m2",
    "prev2_gap_days",
    "prev2_source_deal_date",
    "exact_prev1_price_per_m2",
    "exact_prev1_gap_days",
    "exact_prev1_source_deal_date",
    "exact_prev1_source_area_m2",
    "exact_prev2_price_per_m2",
    "exact_prev2_gap_days",
    "exact_prev2_source_deal_date",
    "exact_prev2_source_area_m2",
}

EXPERIMENTS = [
    {
        "experiment_name": "F10_reference_recheck",
        "numeric_features": WIDE_F10_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 10,
    },
    {
        "experiment_name": "F17_exact_area_prev_replace",
        "numeric_features": EXACT_REPLACE_FEATURES,
        "base_log_feature": "log_exact_prev1_price_per_m2",
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 17,
    },
    {
        "experiment_name": "F18_exact_area_prev_additive",
        "numeric_features": EXACT_ADDITIVE_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 18,
    },
    {
        "experiment_name": "F19_exact_area_prev_fallback_flag",
        "numeric_features": FALLBACK_FEATURES,
        "base_log_feature": "log_fallback_prev1_price_per_m2",
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 19,
    },
]

if EXPERIMENT_FILTER:
    unknown = EXPERIMENT_FILTER - {experiment["experiment_name"] for experiment in EXPERIMENTS}
    if unknown:
        raise SystemExit(f"Unknown E09_EXPERIMENTS values: {sorted(unknown)}")
    EXPERIMENTS = [experiment for experiment in EXPERIMENTS if experiment["experiment_name"] in EXPERIMENT_FILTER]
    if "F10_reference_recheck" not in {experiment["experiment_name"] for experiment in EXPERIMENTS}:
        raise SystemExit("E09_EXPERIMENTS must include F10_reference_recheck for within-run comparison.")


def numeric_features(config: dict) -> list[str]:
    return list(dict.fromkeys(config["numeric_features"]))


def embedding_features(config: dict) -> list[str]:
    return list(config.get("embedding_features", BASE_EMBEDDING_FEATURES))


def numeric_medians_for(config: dict, splits: dict[str, pd.DataFrame]) -> pd.Series:
    return splits["train"][numeric_features(config)].median(numeric_only=True).astype("float32")


def base_log(split_df: pd.DataFrame, medians: pd.Series, config: dict) -> np.ndarray:
    feature = config.get("base_log_feature", "log_complex_prev_price_per_m2")
    return split_df[feature].fillna(medians[feature]).to_numpy(dtype="float32")


def make_inputs(split_df: pd.DataFrame, config: dict, medians: pd.Series) -> dict[str, Any]:
    numeric_df = split_df[numeric_features(config)].copy().fillna(medians)
    inputs = {"numeric_input": numeric_df.to_numpy(dtype="float32")}
    for feature in embedding_features(config):
        values = np.asarray(split_df[feature].fillna("missing").astype("string").astype(str).tolist(), dtype=str).reshape(-1, 1)
        inputs[f"{feature}_input"] = tf.convert_to_tensor(values, dtype=tf.string)
    return inputs


def y_for(split_df: pd.DataFrame, medians: pd.Series, config: dict) -> np.ndarray:
    return split_df["target"].to_numpy(dtype="float32") - base_log(split_df, medians, config)


def final_log_pred(split_df: pd.DataFrame, raw_pred: np.ndarray, medians: pd.Series, config: dict) -> np.ndarray:
    return base_log(split_df, medians, config).astype("float64") + np.asarray(raw_pred, dtype="float64").reshape(-1)


def build_preprocessors(config: dict, train_df: pd.DataFrame, medians: pd.Series):
    train_inputs = make_inputs(train_df, config, medians)
    normalizer = keras.layers.Normalization(name="numeric_normalization")
    normalizer.adapt(train_inputs["numeric_input"])
    lookups = {}
    for feature in embedding_features(config):
        lookup = keras.layers.StringLookup(num_oov_indices=1, mask_token=None, name=f"{feature}_lookup")
        lookup.adapt(train_inputs[f"{feature}_input"])
        lookups[feature] = lookup
    return train_inputs, normalizer, lookups


def build_model(config: dict, normalizer, lookups: dict):
    tf.keras.utils.set_random_seed(RANDOM_STATE + int(config.get("seed_offset", 0)))
    numeric_input = keras.Input(shape=(len(numeric_features(config)),), name="numeric_input", dtype="float32")
    parts = [normalizer(numeric_input)]
    inputs = [numeric_input]
    for feature in embedding_features(config):
        inp = keras.Input(shape=(1,), name=f"{feature}_input", dtype=tf.string)
        idx = lookups[feature](inp)
        dim = int(config["embedding_dims"].get(feature, EMBEDDING_DIMS[feature]))
        emb = keras.layers.Embedding(lookups[feature].vocabulary_size(), dim, name=f"{feature}_embedding")(idx)
        inputs.append(inp)
        parts.append(keras.layers.Flatten(name=f"{feature}_flatten")(emb))
    x = keras.layers.Concatenate(name="feature_concat")(parts)
    for unit in config["dense_units"]:
        x = keras.layers.Dense(unit, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = keras.layers.Dropout(0.10 if unit >= 128 else 0.05)(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


def metric_row(split_df: pd.DataFrame, pred_log: np.ndarray, config: dict, split_name: str) -> dict:
    y_true = split_df["target"].to_numpy(dtype="float64")
    pred_log = np.asarray(pred_log, dtype="float64").reshape(-1)
    pred_ppm = np.exp(pred_log)
    actual_ppm = split_df["price_per_m2"].to_numpy(dtype="float64")
    pred_total = pred_ppm * split_df["area_m2"].to_numpy(dtype="float64")
    actual_total = split_df["price_total"].to_numpy(dtype="float64")
    abs_pct = np.abs((pred_ppm - actual_ppm) / actual_ppm)
    out = {
        "run_mode": RUN_MODE,
        "experiment_name": config["experiment_name"],
        "base_log_feature": config.get("base_log_feature", "log_complex_prev_price_per_m2"),
        "learning_rate": config["learning_rate"],
        "numeric_features": json.dumps(numeric_features(config), ensure_ascii=False),
        "embedding_features": json.dumps(embedding_features(config), ensure_ascii=False),
        "split": split_name,
        "rows": len(split_df),
        "log_mae": float(mean_absolute_error(y_true, pred_log)),
        "log_rmse": float(math.sqrt(mean_squared_error(y_true, pred_log))),
        "price_per_m2_mae": float(mean_absolute_error(actual_ppm, pred_ppm)),
        "price_per_m2_mape": float(np.mean(abs_pct)),
        "total_price_mae_manwon": float(mean_absolute_error(actual_total, pred_total)),
        "abs_pct_error_p95": float(np.quantile(abs_pct, 0.95)),
        "abs_pct_error_p99": float(np.quantile(abs_pct, 0.99)),
    }
    for threshold in ERROR_RATE_THRESHOLDS:
        key = f"error_gt_{int(threshold * 100)}pct_rate"
        out[key] = float((abs_pct > threshold).mean())
        out[key.replace("rate", "rows")] = int((abs_pct > threshold).sum())
    return out


def group_rows(split_df: pd.DataFrame, pred_log: np.ndarray, experiment_name: str, split_name: str) -> list[dict]:
    group_cols = [
        "exact_prev1_missing_group",
        "wide_prev1_present_exact_missing_group",
        "exact_prev1_gap_bucket_plus",
        "prev1_gap_bucket_plus",
        "prev2_gap_bucket_plus",
    ]
    work = split_df[[*group_cols, "target", "price_per_m2"]].copy()
    work["pred_target"] = np.asarray(pred_log, dtype="float64")
    work["abs_log_error"] = (work["pred_target"] - work["target"].astype("float64")).abs()
    work["pred_price_per_m2"] = np.exp(work["pred_target"])
    work["abs_pct_error"] = ((work["pred_price_per_m2"] - work["price_per_m2"].astype("float64")) / work["price_per_m2"].astype("float64")).abs()
    rows = []
    for group_type in group_cols:
        for group_value, group in work.groupby(group_type, dropna=False, observed=True):
            if len(group) < 100:
                continue
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "split": split_name,
                    "group_type": group_type,
                    "group_value": str(group_value),
                    "rows": len(group),
                    "log_mae": float(group["abs_log_error"].mean()),
                    "p95_abs_pct_error": float(group["abs_pct_error"].quantile(0.95)),
                    "p99_abs_pct_error": float(group["abs_pct_error"].quantile(0.99)),
                    "error_gt_20pct_rate": float((group["abs_pct_error"] > 0.20).mean()),
                }
            )
    return rows


def train_and_predict(config: dict, splits: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf.keras.backend.clear_session()
    print("\n===", config["experiment_name"], "===")
    assert not ((set(numeric_features(config)) | set(embedding_features(config))) & HARD_LEAKAGE_COLUMNS)
    medians = numeric_medians_for(config, splits)
    train_inputs, normalizer, lookups = build_preprocessors(config, splits["train"], medians)
    model = build_model(config, normalizer, lookups)
    valid_inputs = make_inputs(splits["valid"], config, medians)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    start = time.perf_counter()
    model.fit(
        train_inputs,
        y_for(splits["train"], medians, config),
        validation_data=(valid_inputs, y_for(splits["valid"], medians, config)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=TRAIN_VERBOSE,
    )
    print("duration_seconds", round(time.perf_counter() - start, 2))
    metrics = []
    groups = []
    for split_name in SPLIT_ORDER:
        inputs = make_inputs(splits[split_name], config, medians)
        raw_pred = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        pred_log = final_log_pred(splits[split_name], raw_pred, medians, config)
        assert np.isfinite(pred_log).all(), (config["experiment_name"], split_name)
        metrics.append(metric_row(splits[split_name], pred_log, config, split_name))
        if split_name in EVAL_SPLITS:
            groups.extend(group_rows(splits[split_name], pred_log, config["experiment_name"], split_name))
    return pd.DataFrame(metrics), pd.DataFrame(groups)


def main() -> int:
    print("python", sys.version)
    print("tensorflow", tf.__version__)
    print("pandas", pd.__version__)
    print("project", PROJECT_DIR)
    print("run_mode", RUN_MODE, "max_epochs", MAX_EPOCHS, "batch_size", BATCH_SIZE)
    ensure_exact_prev_sidecar()

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
    assert exact_df["transaction_id"].is_unique
    assert len(exact_df) == len(raw_df), (len(exact_df), len(raw_df))

    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    assert int(model_df["prev2_missing"].isna().sum()) == 0
    assert int(model_df["exact_prev1_missing"].isna().sum()) == 0
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df = add_model_features(model_df)
    splits = apply_smoke_sampling(split_frames(model_df))
    counts_df = pd.DataFrame([{"split": s, "rows": len(splits[s])} for s in SPLIT_ORDER])
    coverage_df = pd.DataFrame(
        [
            {
                "split": s,
                "rows": len(splits[s]),
                "exact_prev1_missing_rate": float(splits[s]["exact_prev1_missing"].astype("float64").mean()),
                "wide_present_exact_missing_rate": float(splits[s]["wide_prev1_present_exact_missing"].astype("float64").mean()),
            }
            for s in SPLIT_ORDER
        ]
    )
    print(counts_df)
    print(coverage_df)

    metric_frames = []
    group_frames = []
    for config in EXPERIMENTS:
        metrics, groups = train_and_predict(config, splits)
        metric_frames.append(metrics)
        group_frames.append(groups)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    group_metrics_df = pd.concat(group_frames, ignore_index=True)

    reference = metrics_df.loc[metrics_df["experiment_name"] == "F10_reference_recheck", ["split", "log_mae"]].rename(columns={"log_mae": "reference_f10_recheck_log_mae"})
    metrics_df = metrics_df.merge(reference, on="split", how="left")
    metrics_df["reference_e07_f10_log_mae"] = metrics_df["split"].map(REFERENCE_E07_F10_LOG_MAE)
    metrics_df["delta_vs_f10_recheck"] = metrics_df["log_mae"] - metrics_df["reference_f10_recheck_log_mae"]
    metrics_df["delta_vs_e07_f10"] = metrics_df["log_mae"] - metrics_df["reference_e07_f10_log_mae"]
    metrics_df.to_csv(METRICS_PATH, index=False)
    group_metrics_df.to_csv(GROUP_METRICS_PATH, index=False)

    eval_metrics = metrics_df.loc[metrics_df["split"].isin(EVAL_SPLITS)].copy()
    judgement_rows = []
    for name in [e["experiment_name"] for e in EXPERIMENTS if e["experiment_name"] != "F10_reference_recheck"]:
        cand = eval_metrics.loc[eval_metrics["experiment_name"] == name].set_index("split")
        recent_delta = float(cand.loc["recent_holdout", "delta_vs_f10_recheck"])
        test_delta = float(cand.loc["test", "delta_vs_f10_recheck"])
        valid_delta = float(cand.loc["valid", "delta_vs_f10_recheck"])
        p99_delta = float(cand.loc["recent_holdout", "abs_pct_error_p99"] - eval_metrics.loc[(eval_metrics["experiment_name"] == "F10_reference_recheck") & (eval_metrics["split"] == "recent_holdout"), "abs_pct_error_p99"].iloc[0])
        success = recent_delta < -1e-9 and valid_delta <= 0.0005 and test_delta <= 0.0005 and p99_delta <= 0.01
        severe_degrade = recent_delta > 0.002 or test_delta > 0.002 or p99_delta > 0.03
        judgement_rows.append(
            {
                "experiment_name": name,
                "valid_delta_vs_f10_recheck": valid_delta,
                "test_delta_vs_f10_recheck": test_delta,
                "recent_delta_vs_f10_recheck": recent_delta,
                "recent_p99_abs_pct_delta": p99_delta,
                "judgement": "성공" if success else ("실패" if severe_degrade else "보류"),
            }
        )
    judgement_df = pd.DataFrame(judgement_rows)
    overall = "성공" if (judgement_df["judgement"] == "성공").any() else ("실패" if (judgement_df["judgement"] == "실패").all() else "보류")
    pivot = eval_metrics.pivot(index="experiment_name", columns="split", values="log_mae").reset_index()
    tail = eval_metrics[["experiment_name", "split", "abs_pct_error_p95", "abs_pct_error_p99", "error_gt_10pct_rate", "error_gt_20pct_rate", "delta_vs_f10_recheck"]]
    focused_groups = group_metrics_df.loc[
        (group_metrics_df["split"] == "recent_holdout")
        & (
            ((group_metrics_df["group_type"] == "wide_prev1_present_exact_missing_group") & (group_metrics_df["group_value"] == "1"))
            | ((group_metrics_df["group_type"] == "exact_prev1_missing_group") & (group_metrics_df["group_value"].isin(["0", "1"])))
            | ((group_metrics_df["group_type"].isin(["exact_prev1_gap_bucket_plus", "prev1_gap_bucket_plus", "prev2_gap_bucket_plus"])) & (group_metrics_df["group_value"].isin(["366-730", "731+"])))
        )
    ].copy()
    lines = [
        "# E09 exact-area prev feature 실험 요약",
        "",
        "## 1. 결론",
        f"- 결론: `{overall}`",
        "- 비교 기준은 동일 실행의 `F10_reference_recheck`입니다.",
        "- 채택 기준: `recent_holdout log_mae` 개선, valid/test 악화 제한, recent p99 tail 악화 제한.",
        "",
        md_table(judgement_df),
        "",
        "## 2. 실행 설정",
        f"- run_mode: `{RUN_MODE}`",
        f"- max_epochs: `{MAX_EPOCHS}`",
        f"- batch_size: `{BATCH_SIZE}`",
        "- split: `train<=2023`, `valid=2024`, `test=2025`, `recent_holdout>=2026`",
        "- Policy B: `is_cancelled == 0`, `trade_type in [중개거래, unknown]`",
        "- exact-area 기준: `abs(prev_area_m2 - current_area_m2) <= 0.5㎡`",
        "",
        "## 3. Split row 수와 exact coverage",
        md_table(coverage_df),
        "",
        "## 4. 핵심 log_mae",
        md_table(pivot[["experiment_name", "recent_holdout", "test", "valid"]]),
        "",
        "## 5. Tail metrics",
        md_table(tail),
        "",
        "## 6. Focus group metrics",
        md_table(focused_groups),
        "",
        "## 7. 생성 산출물",
        f"- `{EXACT_PREV_PATH}`",
        f"- `{METRICS_PATH}`",
        f"- `{GROUP_METRICS_PATH}`",
        f"- `{SUMMARY_PATH}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("metrics", METRICS_PATH)
    print("group_metrics", GROUP_METRICS_PATH)
    print("summary", SUMMARY_PATH)
    print(judgement_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
