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
NEWS_FEATURE_PATH = PROJECT_DIR / "outputs" / "e08_news_features.csv"
NEWS_BUILDER_PATH = PROJECT_DIR / "scripts" / "build_e08_news_features.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "e08_news_metrics.csv"
GROUP_METRICS_PATH = OUTPUT_DIR / "e08_news_group_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "e08_news_summary.md"

RUN_MODE = os.environ.get("E08_RUN_MODE", "full").strip().lower()
REBUILD_NEWS_FEATURES = os.environ.get("E08_REBUILD_NEWS_FEATURES", "0") == "1"
EXPERIMENT_FILTER = {
    name.strip()
    for name in os.environ.get("E08_EXPERIMENTS", "").split(",")
    if name.strip()
}
RANDOM_STATE = 42
SMOKE_LIMITS = {"train": 200_000, "valid": 50_000, "test": 50_000, "recent_holdout": 50_000}
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]
ERROR_RATE_THRESHOLDS = [0.10, 0.20, 0.30, 0.50]
BATCH_SIZE = int(os.environ.get("E08_BATCH_SIZE", "8192"))
MAX_EPOCHS = int(os.environ.get("E08_MAX_EPOCHS", "30"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("E08_EARLY_STOPPING_PATIENCE", "4"))
TRAIN_VERBOSE = int(os.environ.get("E08_TRAIN_VERBOSE", "2"))
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


def ensure_news_sidecar() -> None:
    if NEWS_FEATURE_PATH.exists() and not REBUILD_NEWS_FEATURES:
        return
    subprocess.run([sys.executable, str(NEWS_BUILDER_PATH)], cwd=PROJECT_DIR, check=True)


def gap_bucket(days: pd.Series) -> pd.Series:
    bucket = pd.Series("missing", index=days.index, dtype="string")
    bucket[(days >= 0) & (days <= 30)] = "0-30"
    bucket[(days >= 31) & (days <= 90)] = "31-90"
    bucket[(days >= 91) & (days <= 180)] = "91-180"
    bucket[(days >= 181) & (days <= 365)] = "181-365"
    bucket[days >= 366] = "366+"
    return bucket.fillna("missing").astype("string")


def add_model_features(input_df: pd.DataFrame) -> pd.DataFrame:
    out = input_df.copy()
    out["target"] = np.log(out["price_per_m2"].astype("float64"))
    out["is_basement_floor"] = (out["floor"] < 0).astype("float32")
    prev1_price = out["complex_prev_price_per_m2"].astype("float64")
    prev2_price = out["complex_prev2_price_per_m2"].astype("float64")
    out["log_complex_prev_price_per_m2"] = np.where(prev1_price > 0, np.log(prev1_price), np.nan)
    out["log_complex_prev2_price_per_m2"] = np.where(prev2_price > 0, np.log(prev2_price), np.nan)
    out["complex_prev_missing"] = out["complex_prev_missing"].fillna(1).astype("float32")
    out["prev2_missing"] = out["prev2_missing"].fillna(1).astype("float32")
    out["prev_deal_gap_months"] = out["prev_deal_gap_days"].astype("float64") / 30.4375
    out["prev2_gap_months"] = out["prev2_gap_days"].astype("float64") / 30.4375
    out["prev1_prev2_gap_months"] = out["prev1_prev2_gap_days"].astype("float64") / 30.4375
    out["prev_deal_gap_bucket"] = gap_bucket(out["prev_deal_gap_days"].astype("float64"))
    out["prev2_missing_group"] = out["prev2_missing"].astype("Int8").astype("string")
    out["parent_news_net_x_prev_trend"] = out["parent_net_signal"].astype("float64") * out["prev1_prev2_log_return"].fillna(0).astype("float64")
    out["national_news_net_x_prev_trend"] = out["national_net_signal"].astype("float64") * out["prev1_prev2_log_return"].fillna(0).astype("float64")
    out["parent_news_net_x_prev2_missing"] = out["parent_net_signal"].astype("float64") * out["prev2_missing"].astype("float64")
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


NUMERIC_F10_FEATURES = [
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
COMPACT_SUFFIXES = [
    "news_confidence",
    "price_up_signal",
    "price_down_signal",
    "net_signal",
    "direct_evidence_count",
    "inherited_evidence_count",
    "matched_news_count_log1p",
    "quality_weight",
    "weighted_net_signal",
    "direct_sufficient_flag",
    "direct_partial_flag",
    "inherited_centered_flag",
    "sparse_flag",
]
TOPIC_SUFFIXES = [
    "policy_positive_score",
    "policy_negative_score",
    "policy_net_signal",
    "redevelopment_score",
    "transport_score",
    "supply_risk_score",
    "sale_market_score",
    "rental_market_score",
]
ROLLING_SUFFIXES = [
    "rolling3_confidence",
    "rolling3_price_up_signal",
    "rolling3_price_down_signal",
    "rolling3_net_signal",
]


def level_features(levels: list[str], suffixes: list[str]) -> list[str]:
    return [f"{level}_{suffix}" for level in levels for suffix in suffixes]


NEWS_DETAIL_COMPACT = level_features(["detail"], COMPACT_SUFFIXES)
NEWS_MULTI_COMPACT = level_features(["detail", "parent", "national"], COMPACT_SUFFIXES)
NEWS_TOPIC_SCORES = level_features(["detail", "parent", "national"], TOPIC_SUFFIXES)
NEWS_ROLLING3 = level_features(["detail", "parent", "national"], ROLLING_SUFFIXES)
NEWS_INTERACTIONS = [
    "detail_parent_net_signal_gap",
    "parent_national_net_signal_gap",
    "detail_parent_confidence_gap",
    "parent_national_confidence_gap",
    "parent_news_net_x_prev_trend",
    "national_news_net_x_prev_trend",
    "parent_news_net_x_prev2_missing",
]
BASE_EMBEDDING_FEATURES = ["legal_dong_code", "sgg_code", "prev_deal_gap_bucket"]
EMBEDDING_DIMS = {"legal_dong_code": 16, "sgg_code": 8, "prev_deal_gap_bucket": 3}
NEWS_GROUP_FEATURES = [
    "detail_news_quality_tier",
    "parent_news_quality_tier",
    "parent_news_bucket",
    "detail_news_bucket",
]
NEWS_REQUIRED_COLUMNS = sorted(
    {
        "transaction_id",
        "news_source_month",
        "news_leakage_violation",
        *NEWS_DETAIL_COMPACT,
        *NEWS_MULTI_COMPACT,
        *NEWS_TOPIC_SCORES,
        *NEWS_ROLLING3,
        *[
            "detail_parent_net_signal_gap",
            "parent_national_net_signal_gap",
            "detail_parent_confidence_gap",
            "parent_national_confidence_gap",
        ],
        *NEWS_GROUP_FEATURES,
    }
)
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
    "news_source_month",
}

EXPERIMENTS = [
    {
        "experiment_name": "F10_reference_recheck",
        "numeric_features": NUMERIC_F10_FEATURES,
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 10,
    },
    {
        "experiment_name": "F11_news_detail_compact",
        "numeric_features": NUMERIC_F10_FEATURES + NEWS_DETAIL_COMPACT,
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 11,
    },
    {
        "experiment_name": "F12_news_multilevel_compact",
        "numeric_features": NUMERIC_F10_FEATURES + NEWS_MULTI_COMPACT,
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 12,
    },
    {
        "experiment_name": "F13_news_topic_scores",
        "numeric_features": NUMERIC_F10_FEATURES + NEWS_MULTI_COMPACT + NEWS_TOPIC_SCORES,
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 13,
    },
    {
        "experiment_name": "F14_news_quality_weighted",
        "numeric_features": NUMERIC_F10_FEATURES + NEWS_MULTI_COMPACT + level_features(["detail", "parent", "national"], ["weighted_net_signal", "quality_weight"]),
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 14,
    },
    {
        "experiment_name": "F15_news_rolling3",
        "numeric_features": NUMERIC_F10_FEATURES + NEWS_MULTI_COMPACT + NEWS_ROLLING3,
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 15,
    },
    {
        "experiment_name": "F16_news_interaction",
        "numeric_features": NUMERIC_F10_FEATURES + NEWS_MULTI_COMPACT + NEWS_ROLLING3 + NEWS_INTERACTIONS,
        "embedding_features": BASE_EMBEDDING_FEATURES,
        "embedding_dims": EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 16,
    },
]

if EXPERIMENT_FILTER:
    unknown = EXPERIMENT_FILTER - {experiment["experiment_name"] for experiment in EXPERIMENTS}
    if unknown:
        raise SystemExit(f"Unknown E08_EXPERIMENTS values: {sorted(unknown)}")
    EXPERIMENTS = [experiment for experiment in EXPERIMENTS if experiment["experiment_name"] in EXPERIMENT_FILTER]
    if "F10_reference_recheck" not in {experiment["experiment_name"] for experiment in EXPERIMENTS}:
        raise SystemExit("E08_EXPERIMENTS must include F10_reference_recheck for within-run comparison.")


def numeric_features(config: dict) -> list[str]:
    return list(dict.fromkeys(config["numeric_features"]))


def embedding_features(config: dict) -> list[str]:
    return list(config.get("embedding_features", BASE_EMBEDDING_FEATURES))


def numeric_medians_for(config: dict, splits: dict[str, pd.DataFrame]) -> pd.Series:
    return splits["train"][numeric_features(config)].median(numeric_only=True).astype("float32")


def base_log(split_df: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return split_df["log_complex_prev_price_per_m2"].fillna(medians["log_complex_prev_price_per_m2"]).to_numpy(dtype="float32")


def make_inputs(split_df: pd.DataFrame, config: dict, medians: pd.Series) -> dict[str, Any]:
    numeric_df = split_df[numeric_features(config)].copy().fillna(medians)
    inputs = {"numeric_input": numeric_df.to_numpy(dtype="float32")}
    for feature in embedding_features(config):
        values = np.asarray(split_df[feature].fillna("missing").astype("string").astype(str).tolist(), dtype=str).reshape(-1, 1)
        inputs[f"{feature}_input"] = tf.convert_to_tensor(values, dtype=tf.string)
    return inputs


def y_for(split_df: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return split_df["target"].to_numpy(dtype="float32") - base_log(split_df, medians)


def final_log_pred(split_df: pd.DataFrame, raw_pred: np.ndarray, medians: pd.Series) -> np.ndarray:
    return base_log(split_df, medians).astype("float64") + np.asarray(raw_pred, dtype="float64").reshape(-1)


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
    abs_log = np.abs(pred_log - y_true)
    abs_pct = np.abs((pred_ppm - actual_ppm) / actual_ppm)
    out = {
        "run_mode": RUN_MODE,
        "experiment_name": config["experiment_name"],
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
    work = split_df[
        [
            "detail_news_quality_tier",
            "parent_news_quality_tier",
            "parent_news_bucket",
            "detail_news_bucket",
            "prev2_missing_group",
            "target",
            "price_per_m2",
        ]
    ].copy()
    work["pred_target"] = np.asarray(pred_log, dtype="float64")
    work["abs_log_error"] = (work["pred_target"] - work["target"].astype("float64")).abs()
    work["pred_price_per_m2"] = np.exp(work["pred_target"])
    work["abs_pct_error"] = ((work["pred_price_per_m2"] - work["price_per_m2"].astype("float64")) / work["price_per_m2"].astype("float64")).abs()
    rows = []
    for group_type in ["detail_news_quality_tier", "parent_news_quality_tier", "parent_news_bucket", "detail_news_bucket", "prev2_missing_group"]:
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
        y_for(splits["train"], medians),
        validation_data=(valid_inputs, y_for(splits["valid"], medians)),
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
        pred_log = final_log_pred(splits[split_name], raw_pred, medians)
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
    ensure_news_sidecar()

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
    news_dtypes = {column: "float32" for column in NEWS_REQUIRED_COLUMNS if column not in {"transaction_id", "news_source_month", "news_leakage_violation", *NEWS_GROUP_FEATURES}}
    news_dtypes.update(
        {
            "transaction_id": "string",
            "news_source_month": "string",
            "news_leakage_violation": "Int8",
            "detail_news_quality_tier": "string",
            "parent_news_quality_tier": "string",
            "parent_news_bucket": "string",
            "detail_news_bucket": "string",
        }
    )
    news_df = pd.read_csv(NEWS_FEATURE_PATH, usecols=NEWS_REQUIRED_COLUMNS, dtype=news_dtypes)
    assert len(news_df) == len(raw_df), (len(news_df), len(raw_df))
    assert news_df["transaction_id"].is_unique
    assert int(news_df["news_leakage_violation"].sum()) == 0

    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(news_df, on="transaction_id", how="left", validate="one_to_one")
    assert int(model_df["prev2_missing"].isna().sum()) == 0
    assert int(model_df["news_source_month"].isna().sum()) == 0
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df = add_model_features(model_df)
    splits = apply_smoke_sampling(split_frames(model_df))
    counts_df = pd.DataFrame([{"split": s, "rows": len(splits[s])} for s in SPLIT_ORDER])
    print(counts_df)

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
    lines = [
        "# E08 news feature 실험 요약",
        "",
        "## 1. 결론",
        f"- 결론: `{overall}`",
        "- 비교 기준은 동일 실행의 `F10_reference_recheck`이며, 보조로 기존 `outputs/e07_prev2_metrics.csv`의 F10 값을 함께 기록합니다.",
        "- 채택 기준: `recent_holdout log_mae` 개선, test 악화 제한, recent p99 tail 악화 제한.",
        "",
        md_table(judgement_df),
        "",
        "## 2. 실행 설정",
        f"- run_mode: `{RUN_MODE}`",
        f"- max_epochs: `{MAX_EPOCHS}`",
        f"- batch_size: `{BATCH_SIZE}`",
        "- split: `train<=2023`, `valid=2024`, `test=2025`, `recent_holdout>=2026`",
        "- Policy B: `is_cancelled == 0`, `trade_type in [중개거래, unknown]`",
        "- news_source_month: `deal_ym - 1 month`",
        "",
        "## 3. Split row 수",
        md_table(counts_df, floatfmt=".0f"),
        "",
        "## 4. 핵심 log_mae",
        md_table(pivot),
        "",
        "## 5. Tail metrics",
        md_table(eval_metrics[["experiment_name", "split", "abs_pct_error_p95", "abs_pct_error_p99", "error_gt_20pct_rate", "delta_vs_f10_recheck", "delta_vs_e07_f10"]]),
        "",
        "## 6. 생성 산출물",
    ]
    for path in [NEWS_FEATURE_PATH, METRICS_PATH, GROUP_METRICS_PATH, SUMMARY_PATH]:
        lines.append(f"- `{path}`")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_PATH)
    print("overall", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
