#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
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
from tensorflow import keras

import build_e11_region_residual_features as e11_builder


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
RESIDUAL_FEATURE_PATH = PROJECT_DIR / "outputs" / "e11_region_residual_features.csv"
RESIDUAL_BUILDER_PATH = PROJECT_DIR / "scripts" / "build_e11_region_residual_features.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "e11_region_residual_metrics.csv"
GROUP_METRICS_PATH = OUTPUT_DIR / "e11_region_residual_group_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "e11_region_residual_summary.md"
EVAL_PREDICTIONS_PATH = OUTPUT_DIR / "e11_f18_eval_predictions.csv"
FINAL_DECISION_PATH = OUTPUT_DIR / "e11_final_decision.md"

RUN_MODE = os.environ.get("E11_RUN_MODE", "full").strip().lower()
REBUILD_RESIDUAL_FEATURES = os.environ.get("E11_REBUILD_RESIDUAL_FEATURES", "0") == "1"
EXPERIMENT_FILTER = {
    name.strip()
    for name in os.environ.get("E11_EXPERIMENTS", "").split(",")
    if name.strip()
}
RANDOM_STATE = 42
BATCH_SIZE = int(os.environ.get("E11_BATCH_SIZE", "8192"))
MAX_EPOCHS = int(os.environ.get("E11_MAX_EPOCHS", "2" if RUN_MODE == "smoke" else "30"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("E11_EARLY_STOPPING_PATIENCE", "2" if RUN_MODE == "smoke" else "4"))
TRAIN_VERBOSE = int(os.environ.get("E11_TRAIN_VERBOSE", "2"))
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]
SMOKE_LIMITS = {"train": 200_000, "valid": 50_000, "test": 50_000, "recent_holdout": 50_000}
ERROR_RATE_THRESHOLDS = [0.10, 0.20]

assert RUN_MODE in {"smoke", "full"}, RUN_MODE
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

e09 = e11_builder.e09
e09.RUN_MODE = RUN_MODE
e09.BATCH_SIZE = BATCH_SIZE
e09.MAX_EPOCHS = MAX_EPOCHS
e09.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
e09.TRAIN_VERBOSE = TRAIN_VERBOSE

F18_FEATURES = list(e09.EXACT_ADDITIVE_FEATURES)
RESIDUAL_BIAS_FEATURES = [
    "complex_resid_bias_log_shrunk",
    "legal_dong_resid_bias_log_shrunk",
    "sgg_resid_bias_log_shrunk",
    "sido_resid_bias_log_shrunk",
    "blended_resid_bias_log",
]
RESIDUAL_RISK_FEATURES = [
    "complex_resid_abs_log_mean",
    "complex_resid_abs_pct_mean",
    "complex_resid_error_gt_10_rate",
    "complex_resid_error_gt_20_rate",
    "complex_resid_count_log1p",
    "complex_resid_confidence",
    "legal_dong_resid_abs_log_mean",
    "legal_dong_resid_abs_pct_mean",
    "legal_dong_resid_error_gt_10_rate",
    "legal_dong_resid_error_gt_20_rate",
    "legal_dong_resid_count_log1p",
    "legal_dong_resid_confidence",
    "sgg_resid_abs_log_mean",
    "sgg_resid_abs_pct_mean",
    "sgg_resid_error_gt_10_rate",
    "sgg_resid_error_gt_20_rate",
    "sgg_resid_count_log1p",
    "sgg_resid_confidence",
    "sido_resid_abs_log_mean",
    "sido_resid_abs_pct_mean",
    "sido_resid_error_gt_10_rate",
    "sido_resid_error_gt_20_rate",
    "sido_resid_count_log1p",
    "sido_resid_confidence",
    "best_resid_source_count_log1p",
    "best_resid_abs_pct_mean",
    "resid_expected_abs_pct_error",
]

EXPERIMENTS = [
    {
        "experiment_name": "F18_reference_recheck",
        "kind": "model",
        "numeric_features": F18_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 18,
        "loss": "mse",
    },
    {
        "experiment_name": "F25_sgg_bias_calibration",
        "kind": "calibration",
        "bias_feature": "sgg_resid_bias_log_shrunk",
        "calibration_policy": "raw_f18_plus_sgg_residual_bias",
    },
    {
        "experiment_name": "F26_multilevel_bias_calibration",
        "kind": "calibration",
        "bias_feature": "blended_resid_bias_log",
        "calibration_policy": "raw_f18_plus_blended_residual_bias",
    },
    {
        "experiment_name": "F29_residual_bias_features_huber",
        "kind": "model",
        "numeric_features": F18_FEATURES + RESIDUAL_RISK_FEATURES + RESIDUAL_BIAS_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 29,
        "loss": "huber_005",
    },
    {
        "experiment_name": "F30_confidence_only_policy",
        "kind": "confidence_only",
        "calibration_policy": "raw_f18_price_unchanged_confidence_only",
    },
]

if EXPERIMENT_FILTER:
    known = {experiment["experiment_name"] for experiment in EXPERIMENTS}
    unknown = EXPERIMENT_FILTER - known
    if unknown:
        raise SystemExit(f"Unknown E11_EXPERIMENTS values: {sorted(unknown)}")
    EXPERIMENTS = [experiment for experiment in EXPERIMENTS if experiment["experiment_name"] in EXPERIMENT_FILTER]
    if "F18_reference_recheck" not in {experiment["experiment_name"] for experiment in EXPERIMENTS}:
        raise SystemExit("E11_EXPERIMENTS must include F18_reference_recheck for within-run comparison.")


def md_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float", "float32", "float64"]).columns:
        x[col] = x[col].map(lambda v: format(v, floatfmt) if pd.notna(v) else "")
    x = x.astype("string").fillna("")
    lines = ["| " + " | ".join(x.columns) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in x.values.tolist()]
    return "\n".join(lines)


def ensure_residual_sidecar() -> None:
    required = [
        RESIDUAL_FEATURE_PATH,
        e11_builder.OOF_PREDICTIONS_PATH,
        e11_builder.CONFIDENCE_REPORT_PATH,
        e11_builder.POLICY_PATH,
        e11_builder.QUALITY_REPORT_PATH,
    ]
    if all(path.exists() for path in required) and not REBUILD_RESIDUAL_FEATURES:
        return
    print("build_residual_sidecar", RESIDUAL_BUILDER_PATH)
    e11_builder.main()


def build_model(config: dict, normalizer, lookups: dict):
    tf.keras.utils.set_random_seed(RANDOM_STATE + int(config.get("seed_offset", 0)))
    numeric_input = keras.Input(shape=(len(e09.numeric_features(config)),), name="numeric_input", dtype="float32")
    parts = [normalizer(numeric_input)]
    inputs = [numeric_input]
    for feature in e09.embedding_features(config):
        inp = keras.Input(shape=(1,), name=f"{feature}_input", dtype=tf.string)
        idx = lookups[feature](inp)
        dim = int(config["embedding_dims"].get(feature, e09.EMBEDDING_DIMS[feature]))
        emb = keras.layers.Embedding(lookups[feature].vocabulary_size(), dim, name=f"{feature}_embedding")(idx)
        inputs.append(inp)
        parts.append(keras.layers.Flatten(name=f"{feature}_flatten")(emb))
    x = keras.layers.Concatenate(name="feature_concat")(parts)
    for unit in config["dense_units"]:
        x = keras.layers.Dense(unit, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = keras.layers.Dropout(0.10 if unit >= 128 else 0.05)(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    loss_name = config.get("loss", "mse")
    if loss_name == "huber_005":
        loss: Any = keras.losses.Huber(delta=0.05)
    elif loss_name == "huber_010":
        loss = keras.losses.Huber(delta=0.10)
    elif loss_name == "logcosh":
        loss = keras.losses.LogCosh()
    else:
        loss = "mse"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss=loss, metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


def split_frames(policy_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    splits = {
        "train": policy_df.loc[policy_df["deal_date"] <= "2023-12-31"],
        "valid": policy_df.loc[(policy_df["deal_date"] >= "2024-01-01") & (policy_df["deal_date"] <= "2024-12-31")],
        "test": policy_df.loc[(policy_df["deal_date"] >= "2025-01-01") & (policy_df["deal_date"] <= "2025-12-31")],
        "recent_holdout": policy_df.loc[policy_df["deal_date"] >= "2026-01-01"],
    }
    for name, frame in splits.items():
        if len(frame) == 0:
            raise RuntimeError(f"empty split: {name}")
    return splits


def apply_smoke_sampling(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if RUN_MODE != "smoke":
        return {key: value.copy() for key, value in splits.items()}
    out = {}
    for name, frame in splits.items():
        limit = SMOKE_LIMITS[name]
        out[name] = frame.sample(n=limit, random_state=RANDOM_STATE).sort_values(["deal_date", "transaction_id"]) if len(frame) > limit else frame.copy()
    return out


def confidence_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype("float64").fillna(0),
        bins=[-np.inf, 0.0, 0.2, 0.5, 0.8, np.inf],
        labels=["0", "0-0.2", "0.2-0.5", "0.5-0.8", "0.8+"],
    ).astype("string")


def load_model_frame() -> pd.DataFrame:
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
        "deal_ym": "string",
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
    residual_df = pd.read_csv(
        RESIDUAL_FEATURE_PATH,
        dtype={
            "transaction_id": "string",
            "resid_source_until_ym": "string",
            "resid_risk_tier": "string",
            "best_resid_level": "string",
        },
    )
    for label, frame in [("raw", raw_df), ("prev2", prev2_df), ("exact", exact_df), ("residual", residual_df)]:
        if not frame["transaction_id"].is_unique:
            raise RuntimeError(f"{label} transaction_id is not unique")
    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(residual_df, on="transaction_id", how="left", validate="one_to_one")
    for col in ["prev2_missing", "exact_prev1_missing", "resid_source_until_ym"]:
        if int(model_df[col].isna().sum()) != 0:
            raise RuntimeError(f"sidecar join missing: {col}")
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df["sido_code"] = model_df["sgg_code"].fillna("missing").astype("string").str.slice(0, 2)
    model_df = e09.add_model_features(model_df)
    for col in RESIDUAL_BIAS_FEATURES + RESIDUAL_RISK_FEATURES:
        if col == "best_resid_source_count_log1p":
            model_df[col] = np.log1p(model_df["best_resid_source_count"].astype("float64").fillna(0)).astype("float32")
            continue
        if col not in model_df.columns:
            raise RuntimeError(f"missing E11 residual feature: {col}")
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    model_df["resid_risk_tier"] = model_df["resid_risk_tier"].fillna("unknown").astype("string")
    model_df["complex_resid_confidence_bucket"] = confidence_bucket(model_df["complex_resid_confidence"])
    model_df["sido_code"] = model_df["sido_code"].fillna("missing").astype("string")
    return model_df.sort_values(["deal_date", "transaction_id"]).reset_index(drop=True)


def metric_row(split_df: pd.DataFrame, pred_log: np.ndarray, experiment_name: str, split_name: str) -> dict:
    y_true = split_df["target"].to_numpy(dtype="float64")
    pred_log = np.asarray(pred_log, dtype="float64").reshape(-1)
    pred_ppm = np.exp(pred_log)
    actual_ppm = split_df["price_per_m2"].to_numpy(dtype="float64")
    abs_pct = np.abs((pred_ppm - actual_ppm) / actual_ppm)
    abs_log = np.abs(pred_log - y_true)
    return {
        "run_mode": RUN_MODE,
        "experiment_name": experiment_name,
        "split": split_name,
        "rows": len(split_df),
        "log_mae": float(abs_log.mean()),
        "price_per_m2_mape": float(abs_pct.mean()),
        "abs_pct_error_p50": float(np.quantile(abs_pct, 0.50)),
        "abs_pct_error_p80": float(np.quantile(abs_pct, 0.80)),
        "abs_pct_error_p90": float(np.quantile(abs_pct, 0.90)),
        "abs_pct_error_p95": float(np.quantile(abs_pct, 0.95)),
        "abs_pct_error_p99": float(np.quantile(abs_pct, 0.99)),
        "error_gt_10pct_rate": float((abs_pct > 0.10).mean()),
        "error_gt_20pct_rate": float((abs_pct > 0.20).mean()),
    }


def group_rows(split_df: pd.DataFrame, pred_log: np.ndarray, experiment_name: str, split_name: str) -> list[dict]:
    group_cols = [
        "resid_risk_tier",
        "sgg_code",
        "sido_code",
        "exact_prev1_missing_group",
        "prev1_gap_bucket_plus",
        "complex_resid_confidence_bucket",
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
                    "median_abs_pct_error": float(group["abs_pct_error"].median()),
                    "p95_abs_pct_error": float(group["abs_pct_error"].quantile(0.95)),
                    "p99_abs_pct_error": float(group["abs_pct_error"].quantile(0.99)),
                    "error_gt_20pct_rate": float((group["abs_pct_error"] > 0.20).mean()),
                }
            )
    return rows


def train_model_predictions(config: dict, splits: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    tf.keras.backend.clear_session()
    print("\n===", config["experiment_name"], "===")
    medians = e09.numeric_medians_for(config, splits)
    train_inputs, normalizer, lookups = e09.build_preprocessors(config, splits["train"], medians)
    model = build_model(config, normalizer, lookups)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    start = time.perf_counter()
    model.fit(
        train_inputs,
        e09.y_for(splits["train"], medians, config),
        validation_data=(e09.make_inputs(splits["valid"], config, medians), e09.y_for(splits["valid"], medians, config)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=TRAIN_VERBOSE,
    )
    print("duration_seconds", round(time.perf_counter() - start, 2))
    preds = {}
    for split_name in SPLIT_ORDER:
        raw_pred = model.predict(e09.make_inputs(splits[split_name], config, medians), batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        pred_log = e09.final_log_pred(splits[split_name], raw_pred, medians, config)
        if not np.isfinite(pred_log).all():
            raise RuntimeError(f"non-finite prediction: {config['experiment_name']} {split_name}")
        preds[split_name] = pred_log
    return preds


def metrics_for_predictions(predictions: dict[str, dict[str, np.ndarray]], splits: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    group_metric_rows = []
    for experiment_name, split_preds in predictions.items():
        for split_name in SPLIT_ORDER:
            metric_rows.append(metric_row(splits[split_name], split_preds[split_name], experiment_name, split_name))
            if split_name in EVAL_SPLITS:
                group_metric_rows.extend(group_rows(splits[split_name], split_preds[split_name], experiment_name, split_name))
    metrics_df = pd.DataFrame(metric_rows)
    reference = metrics_df.loc[metrics_df["experiment_name"] == "F18_reference_recheck", ["split", "log_mae", "error_gt_10pct_rate", "error_gt_20pct_rate", "abs_pct_error_p99"]].rename(
        columns={
            "log_mae": "f18_log_mae",
            "error_gt_10pct_rate": "f18_error_gt_10pct_rate",
            "error_gt_20pct_rate": "f18_error_gt_20pct_rate",
            "abs_pct_error_p99": "f18_abs_pct_error_p99",
        }
    )
    metrics_df = metrics_df.merge(reference, on="split", how="left")
    metrics_df["delta_vs_f18_log_mae"] = metrics_df["log_mae"] - metrics_df["f18_log_mae"]
    metrics_df["delta_vs_f18_gt10"] = metrics_df["error_gt_10pct_rate"] - metrics_df["f18_error_gt_10pct_rate"]
    metrics_df["delta_vs_f18_gt20"] = metrics_df["error_gt_20pct_rate"] - metrics_df["f18_error_gt_20pct_rate"]
    metrics_df["delta_vs_f18_p99"] = metrics_df["abs_pct_error_p99"] - metrics_df["f18_abs_pct_error_p99"]
    drop_cols = ["f18_log_mae", "f18_error_gt_10pct_rate", "f18_error_gt_20pct_rate", "f18_abs_pct_error_p99"]
    metrics_df = metrics_df.drop(columns=drop_cols)
    return metrics_df, pd.DataFrame(group_metric_rows)


def judgement(metrics_df: pd.DataFrame) -> pd.DataFrame:
    eval_metrics = metrics_df.loc[metrics_df["split"].isin(EVAL_SPLITS)].copy()
    rows = []
    for name in [experiment["experiment_name"] for experiment in EXPERIMENTS if experiment["experiment_name"] != "F18_reference_recheck"]:
        cand = eval_metrics.loc[eval_metrics["experiment_name"] == name].set_index("split")
        if cand.empty:
            continue
        recent_delta = float(cand.loc["recent_holdout", "delta_vs_f18_log_mae"])
        test_delta = float(cand.loc["test", "delta_vs_f18_log_mae"])
        valid_delta = float(cand.loc["valid", "delta_vs_f18_log_mae"])
        p99_delta = float(cand.loc["recent_holdout", "delta_vs_f18_p99"])
        gt20_delta = float(cand.loc["recent_holdout", "delta_vs_f18_gt20"])
        is_confidence_only = name == "F30_confidence_only_policy"
        success = (not is_confidence_only) and recent_delta < -1e-9 and test_delta <= 0.0005 and p99_delta <= 0.01 and gt20_delta <= 0.002
        severe_degrade = (not is_confidence_only) and (recent_delta > 0.002 or test_delta > 0.002 or p99_delta > 0.03 or gt20_delta > 0.005)
        rows.append(
            {
                "experiment_name": name,
                "valid_delta_vs_f18_log_mae": valid_delta,
                "test_delta_vs_f18_log_mae": test_delta,
                "recent_delta_vs_f18_log_mae": recent_delta,
                "recent_p99_abs_pct_delta": p99_delta,
                "recent_gt20_rate_delta": gt20_delta,
                "judgement": "보조 유지" if is_confidence_only else ("성공" if success else ("실패" if severe_degrade else "보류")),
            }
        )
    return pd.DataFrame(rows)


def choose_candidates(judgement_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict[str, str]:
    successful = judgement_df.loc[judgement_df["judgement"].eq("성공"), "experiment_name"].tolist()
    price_names = [name for name in successful if name in {"F25_sgg_bias_calibration", "F26_multilevel_bias_calibration"}]
    feature_names = [name for name in successful if name in {"F29_residual_bias_features_huber"}]

    def best(names: list[str]) -> str:
        if not names:
            return ""
        recent = metrics_df.loc[(metrics_df["split"] == "recent_holdout") & (metrics_df["experiment_name"].isin(names))]
        return str(recent.sort_values(["log_mae", "abs_pct_error_p99"]).iloc[0]["experiment_name"])

    best_price = best(price_names)
    best_feature = best(feature_names)
    all_success = [name for name in [best_price, best_feature] if name]
    final_model = best(all_success) if all_success else "F18_reference_recheck"
    return {
        "best_price_calibration": best_price,
        "best_feature_model": best_feature,
        "final_model": final_model,
        "confidence_policy": "F30_confidence_only_policy",
    }


def prediction_error_columns(actual_ppm: np.ndarray, pred_ppm: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    signed = (pred_ppm - actual_ppm) / actual_ppm
    return {
        f"{prefix}_signed_pct_error": signed,
        f"{prefix}_abs_pct_error": np.abs(signed),
    }


def write_eval_predictions(predictions: dict[str, dict[str, np.ndarray]], splits: dict[str, pd.DataFrame], choices: dict[str, str]) -> None:
    calibrated_name = "F26_multilevel_bias_calibration" if "F26_multilevel_bias_calibration" in predictions else ("F25_sgg_bias_calibration" if "F25_sgg_bias_calibration" in predictions else "F18_reference_recheck")
    feature_name = choices.get("best_feature_model") or ("F29_residual_bias_features_huber" if "F29_residual_bias_features_huber" in predictions else "F18_reference_recheck")
    frames = []
    for split_name in EVAL_SPLITS:
        frame = splits[split_name]
        actual_ppm = frame["price_per_m2"].to_numpy(dtype="float64")
        raw_ppm = np.exp(predictions["F18_reference_recheck"][split_name])
        calibrated_ppm = np.exp(predictions[calibrated_name][split_name])
        feature_ppm = np.exp(predictions[feature_name][split_name])
        out = frame[["transaction_id", "resid_risk_tier"]].copy()
        out.insert(1, "split", split_name)
        out["actual_price_per_m2"] = actual_ppm
        out["raw_f18_pred_price_per_m2"] = raw_ppm
        out["calibrated_pred_price_per_m2"] = calibrated_ppm
        out["feature_model_pred_price_per_m2"] = feature_ppm
        for key, value in prediction_error_columns(actual_ppm, raw_ppm, "raw_f18").items():
            out[key] = value
        for key, value in prediction_error_columns(actual_ppm, calibrated_ppm, "calibrated").items():
            out[key] = value
        for key, value in prediction_error_columns(actual_ppm, feature_ppm, "feature_model").items():
            out[key] = value
        out["calibration_policy"] = calibrated_name
        out["feature_model_policy"] = feature_name
        frames.append(out)
    pd.concat(frames, ignore_index=True).to_csv(EVAL_PREDICTIONS_PATH, index=False)


def write_reports(metrics_df: pd.DataFrame, group_metrics_df: pd.DataFrame, judgement_df: pd.DataFrame, choices: dict[str, str], splits: dict[str, pd.DataFrame]) -> None:
    eval_metrics = metrics_df.loc[metrics_df["split"].isin(EVAL_SPLITS)].copy()
    pivot = eval_metrics.pivot(index="experiment_name", columns="split", values="log_mae").reset_index()
    tail = eval_metrics[
        [
            "experiment_name",
            "split",
            "abs_pct_error_p90",
            "abs_pct_error_p95",
            "abs_pct_error_p99",
            "error_gt_10pct_rate",
            "error_gt_20pct_rate",
            "delta_vs_f18_log_mae",
            "delta_vs_f18_gt20",
            "delta_vs_f18_p99",
        ]
    ]
    split_coverage = pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": len(frame),
                "complex_resid_source_rate": float(frame["complex_resid_source_count"].gt(0).mean()),
                "sgg_resid_source_rate": float(frame["sgg_resid_source_count"].gt(0).mean()),
                "risk_unknown_rate": float(frame["resid_risk_tier"].eq("unknown").mean()),
                "risk_high_rate": float(frame["resid_risk_tier"].eq("high").mean()),
            }
            for split_name, frame in splits.items()
        ]
    )
    focused_groups = group_metrics_df.loc[
        (group_metrics_df["split"] == "recent_holdout")
        & (
            ((group_metrics_df["group_type"] == "resid_risk_tier") & (group_metrics_df["group_value"].isin(["unknown", "high"])))
            | ((group_metrics_df["group_type"] == "complex_resid_confidence_bucket") & (group_metrics_df["group_value"].isin(["0", "0-0.2"])))
            | ((group_metrics_df["group_type"] == "prev1_gap_bucket_plus") & (group_metrics_df["group_value"].isin(["366-730", "731+"])))
        )
    ].copy()
    overall = "성공" if (judgement_df["judgement"] == "성공").any() else ("보류" if (judgement_df["judgement"].isin(["보류", "보조 유지"])).any() else "실패")
    lines = [
        "# E11 region residual calibration / feature 실험 요약",
        "",
        "## 1. 결론",
        f"- 결론: `{overall}`",
        f"- 최종 가격 후보: `{choices['final_model']}`",
        f"- 가격 보정 후보: `{choices['best_price_calibration'] or '미채택'}`",
        f"- residual feature 후보: `{choices['best_feature_model'] or '미채택'}`",
        "- `F30_confidence_only_policy`는 가격 성능 후보가 아니라 실사용 신뢰도/예상범위 보조 산출물입니다.",
        "",
        md_table(judgement_df),
        "",
        "## 2. 실행 설정",
        f"- run_mode: `{RUN_MODE}`",
        f"- max_epochs: `{MAX_EPOCHS}`",
        f"- batch_size: `{BATCH_SIZE}`",
        "- split: `train<=2023`, `valid=2024`, `test=2025`, `recent_holdout>=2026`",
        "- Policy B: `is_cancelled == 0`, `trade_type in [중개거래, unknown]`",
        "- residual source: valid `<=2023`, test `<=2024`, recent `<=2025`, train은 row별 이전 월만 사용합니다.",
        "",
        "## 3. Residual coverage",
        md_table(split_coverage),
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
        f"- `{e11_builder.OOF_PREDICTIONS_PATH}`",
        f"- `{RESIDUAL_FEATURE_PATH}`",
        f"- `{METRICS_PATH}`",
        f"- `{GROUP_METRICS_PATH}`",
        f"- `{EVAL_PREDICTIONS_PATH}`",
        f"- `{e11_builder.CONFIDENCE_REPORT_PATH}`",
        f"- `{e11_builder.POLICY_PATH}`",
        f"- `{SUMMARY_PATH}`",
        f"- `{FINAL_DECISION_PATH}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    final_recent = metrics_df.loc[(metrics_df["experiment_name"] == choices["final_model"]) & (metrics_df["split"] == "recent_holdout")].iloc[0]
    decision_lines = [
        "# E11 final decision",
        "",
        "## 최종 채택",
        f"- 최종 채택 모델: `{choices['final_model']}`",
        f"- recent_holdout log_mae: `{float(final_recent['log_mae']):.6f}`",
        f"- raw F18 대비 recent_holdout log_mae delta: `{float(final_recent['delta_vs_f18_log_mae']):.6f}`",
        f"- raw F18 대비 recent p99 delta: `{float(final_recent['delta_vs_f18_p99']):.6f}`",
        "",
        "## 채택 여부",
        f"- 가격 보정 채택 여부: `{'채택' if choices['best_price_calibration'] else '미채택'}`",
        f"- feature 채택 여부: `{'채택' if choices['best_feature_model'] else '미채택'}`",
        "- 신뢰도 정책 채택 여부: `채택`",
        "",
        "## 발표/보고용 한 줄 결론",
        f"- F18 기준 예측을 유지/개선 후보와 비교하되, 지역 residual 이력은 가격 보정 채택 여부와 무관하게 신뢰도와 예상 오차 범위 설명값으로 유지합니다.",
        "",
        "## 검증 근거 확인",
        f"- metrics_csv: `{METRICS_PATH}`",
        f"- group_metrics_csv: `{GROUP_METRICS_PATH}`",
        f"- eval_predictions_csv: `{EVAL_PREDICTIONS_PATH}`",
        f"- confidence_report_csv: `{e11_builder.CONFIDENCE_REPORT_PATH}`",
    ]
    FINAL_DECISION_PATH.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("python", sys.version)
    print("tensorflow", tf.__version__)
    print("pandas", pd.__version__)
    print("project", PROJECT_DIR)
    print("run_mode", RUN_MODE, "max_epochs", MAX_EPOCHS, "batch_size", BATCH_SIZE)
    ensure_residual_sidecar()
    model_df = load_model_frame()
    splits = apply_smoke_sampling(split_frames(model_df))
    print(pd.DataFrame([{"split": name, "rows": len(frame)} for name, frame in splits.items()]))

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for config in EXPERIMENTS:
        name = config["experiment_name"]
        kind = config["kind"]
        if kind == "model":
            predictions[name] = train_model_predictions(config, splits)
        elif kind == "calibration":
            if "F18_reference_recheck" not in predictions:
                raise RuntimeError(f"{name} requires F18_reference_recheck predictions first")
            bias_feature = config["bias_feature"]
            predictions[name] = {
                split_name: predictions["F18_reference_recheck"][split_name] + splits[split_name][bias_feature].to_numpy(dtype="float64")
                for split_name in SPLIT_ORDER
            }
        elif kind == "confidence_only":
            if "F18_reference_recheck" not in predictions:
                raise RuntimeError(f"{name} requires F18_reference_recheck predictions first")
            predictions[name] = {split_name: values.copy() for split_name, values in predictions["F18_reference_recheck"].items()}
        else:
            raise RuntimeError(f"unknown experiment kind: {kind}")

    metrics_df, group_metrics_df = metrics_for_predictions(predictions, splits)
    judgement_df = judgement(metrics_df)
    choices = choose_candidates(judgement_df, metrics_df)
    write_eval_predictions(predictions, splits, choices)
    metrics_df.to_csv(METRICS_PATH, index=False)
    group_metrics_df.to_csv(GROUP_METRICS_PATH, index=False)
    write_reports(metrics_df, group_metrics_df, judgement_df, choices, splits)
    print("metrics", METRICS_PATH)
    print("group_metrics", GROUP_METRICS_PATH)
    print("eval_predictions", EVAL_PREDICTIONS_PATH)
    print("summary", SUMMARY_PATH)
    print("final_decision", FINAL_DECISION_PATH)
    print(judgement_df)
    print("seconds", round(time.perf_counter() - start, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
