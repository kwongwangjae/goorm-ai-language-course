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
from sklearn.metrics import mean_absolute_error
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
OUTLIER_PATH = PROJECT_DIR / "outputs" / "e10_outlier_signal_features.csv"
OUTLIER_BUILDER_PATH = PROJECT_DIR / "scripts" / "build_e10_outlier_signal_features.py"
E09_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_e09_exact_prev_experiments.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "e10_outlier_signal_metrics.csv"
GROUP_METRICS_PATH = OUTPUT_DIR / "e10_outlier_signal_group_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "e10_outlier_signal_summary.md"

RUN_MODE = os.environ.get("E10_RUN_MODE", "full").strip().lower()
REBUILD_OUTLIER_FEATURES = os.environ.get("E10_REBUILD_OUTLIER_FEATURES", "0") == "1"
EXPERIMENT_FILTER = {
    name.strip()
    for name in os.environ.get("E10_EXPERIMENTS", "").split(",")
    if name.strip()
}
RANDOM_STATE = 42
SMOKE_LIMITS = {"train": 200_000, "valid": 50_000, "test": 50_000, "recent_holdout": 50_000}
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]
ERROR_RATE_THRESHOLDS = [0.10, 0.20, 0.30, 0.50]
BATCH_SIZE = int(os.environ.get("E10_BATCH_SIZE", "8192"))
MAX_EPOCHS = int(os.environ.get("E10_MAX_EPOCHS", "30"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("E10_EARLY_STOPPING_PATIENCE", "4"))
TRAIN_VERBOSE = int(os.environ.get("E10_TRAIN_VERBOSE", "2"))

assert RUN_MODE in {"smoke", "full"}, RUN_MODE
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)


def load_e09_module():
    spec = importlib.util.spec_from_file_location("e09_exact_prev_runner", E09_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E09_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN_MODE = RUN_MODE
    module.BATCH_SIZE = BATCH_SIZE
    module.MAX_EPOCHS = MAX_EPOCHS
    module.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
    module.TRAIN_VERBOSE = TRAIN_VERBOSE
    return module


e09 = load_e09_module()


def md_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float", "float32", "float64"]).columns:
        x[col] = x[col].map(lambda v: format(v, floatfmt) if pd.notna(v) else "")
    x = x.astype("string").fillna("")
    lines = ["| " + " | ".join(x.columns) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in x.values.tolist()]
    return "\n".join(lines)


def ensure_outlier_sidecar() -> None:
    if OUTLIER_PATH.exists() and not REBUILD_OUTLIER_FEATURES:
        return
    spec = importlib.util.spec_from_file_location("e10_outlier_builder", OUTLIER_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {OUTLIER_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


OUTLIER_BASIC_FEATURES = [
    "wide_prev_jump_signed",
    "wide_prev_jump_abs",
    "wide_prev_jump_up",
    "wide_prev_jump_down_abs",
    "is_wide_prev_jump_10pct",
    "is_wide_prev_jump_20pct",
    "is_wide_prev_jump_30pct",
    "is_wide_prev_jump_up_20pct",
    "is_wide_prev_jump_down_20pct",
    "exact_prev_jump_signed",
    "exact_prev_jump_abs",
    "exact_prev_jump_up",
    "exact_prev_jump_down_abs",
    "is_exact_prev_jump_10pct",
    "is_exact_prev_jump_20pct",
    "is_exact_prev_jump_30pct",
    "is_exact_prev_jump_up_20pct",
    "is_exact_prev_jump_down_20pct",
    "exact_wide_prev1_log_gap",
    "exact_wide_prev1_log_gap_abs",
    "is_exact_wide_prev1_gap_5pct",
    "is_exact_wide_prev1_gap_10pct",
    "is_exact_wide_prev1_gap_20pct",
    "wide_prev_jump_abs_x_gap_log1p",
    "exact_prev_jump_abs_x_gap_log1p",
    "is_prev_gap_365d",
    "is_prev_gap_730d",
    "is_exact_prev_gap_365d",
    "is_exact_prev_gap_730d",
    "is_wide_jump20_and_gap365",
    "is_exact_jump20_and_gap365",
    "outlier_signal_score",
]
REGION_PRIOR_FEATURES = [
    "sgg_lag1_log_median_ppm",
    "sgg_lag1_month_count_log1p",
    "sgg_roll3_log_median_ppm",
    "sgg_roll6_log_median_ppm",
    "sgg_prior_missing",
    "wide_prev1_vs_sgg_lag1_log_gap",
    "wide_prev1_vs_sgg_lag1_log_gap_abs",
    "exact_prev1_vs_sgg_lag1_log_gap",
    "exact_prev1_vs_sgg_lag1_log_gap_abs",
    "is_wide_prev1_region_outlier_10pct",
    "is_wide_prev1_region_outlier_20pct",
    "is_wide_prev1_region_outlier_30pct",
    "is_exact_prev1_region_outlier_10pct",
    "is_exact_prev1_region_outlier_20pct",
    "is_exact_prev1_region_outlier_30pct",
]

F18_FEATURES = list(e09.EXACT_ADDITIVE_FEATURES)
OUTLIER_MINIMAL_FEATURES = [
    "is_wide_prev_jump_20pct",
    "is_exact_prev_jump_20pct",
    "is_exact_wide_prev1_gap_10pct",
    "is_wide_jump20_and_gap365",
    "is_exact_jump20_and_gap365",
    "outlier_signal_score",
]
F19_FEATURES = F18_FEATURES + OUTLIER_MINIMAL_FEATURES
F20_FEATURES = F18_FEATURES + OUTLIER_BASIC_FEATURES
F21_FEATURES = F20_FEATURES + REGION_PRIOR_FEATURES

EXPERIMENTS = [
    {
        "experiment_name": "F18_reference_recheck",
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
        "experiment_name": "F19_outlier_signal_minimal",
        "numeric_features": F19_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 19,
        "loss": "mse",
    },
    {
        "experiment_name": "F19_outlier_signal_minimal_huber",
        "numeric_features": F19_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 191,
        "loss": "huber_005",
    },
    {
        "experiment_name": "F18_reference_huber",
        "numeric_features": F18_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 182,
        "loss": "huber_005",
    },
    {
        "experiment_name": "F18_reference_huber_010",
        "numeric_features": F18_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 183,
        "loss": "huber_010",
    },
    {
        "experiment_name": "F18_reference_logcosh",
        "numeric_features": F18_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 184,
        "loss": "logcosh",
    },
    {
        "experiment_name": "F20_outlier_signal_basic",
        "numeric_features": F20_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 20,
        "loss": "mse",
    },
    {
        "experiment_name": "F21_outlier_signal_region_prior",
        "numeric_features": F21_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 21,
        "loss": "mse",
    },
    {
        "experiment_name": "F22_outlier_signal_region_huber",
        "numeric_features": F21_FEATURES,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": 22,
        "loss": "huber_005",
    },
]

if EXPERIMENT_FILTER:
    unknown = EXPERIMENT_FILTER - {experiment["experiment_name"] for experiment in EXPERIMENTS}
    if unknown:
        raise SystemExit(f"Unknown E10_EXPERIMENTS values: {sorted(unknown)}")
    EXPERIMENTS = [experiment for experiment in EXPERIMENTS if experiment["experiment_name"] in EXPERIMENT_FILTER]
    if "F18_reference_recheck" not in {experiment["experiment_name"] for experiment in EXPERIMENTS}:
        raise SystemExit("E10_EXPERIMENTS must include F18_reference_recheck for within-run comparison.")


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


def group_rows(split_df: pd.DataFrame, pred_log: np.ndarray, experiment_name: str, split_name: str) -> list[dict]:
    group_cols = [
        "exact_prev1_missing_group",
        "wide_prev1_present_exact_missing_group",
        "exact_prev1_gap_bucket_plus",
        "prev1_gap_bucket_plus",
        "prev2_gap_bucket_plus",
        "wide_prev_jump_20_group",
        "exact_prev_jump_20_group",
        "exact_wide_gap_10_group",
        "region_prior_missing_group",
        "wide_region_outlier_20_group",
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


e09.build_model = build_model
e09.group_rows = group_rows


def add_e10_features(model_df: pd.DataFrame) -> pd.DataFrame:
    out = e09.add_model_features(model_df)
    numeric_cols = OUTLIER_BASIC_FEATURES + REGION_PRIOR_FEATURES
    for col in numeric_cols:
        if col not in out.columns:
            raise RuntimeError(f"missing E10 feature column: {col}")
        if col.endswith("_missing"):
            out[col] = out[col].fillna(1).astype("float32")
        elif col.startswith("sgg_") and "log_median" in col:
            out[col] = out[col].astype("float32")
        else:
            out[col] = out[col].fillna(0).astype("float32")
    out["wide_prev_jump_20_group"] = out["is_wide_prev_jump_20pct"].astype("Int8").astype("string")
    out["exact_prev_jump_20_group"] = out["is_exact_prev_jump_20pct"].astype("Int8").astype("string")
    out["exact_wide_gap_10_group"] = out["is_exact_wide_prev1_gap_10pct"].astype("Int8").astype("string")
    out["region_prior_missing_group"] = out["sgg_prior_missing"].astype("Int8").astype("string")
    out["wide_region_outlier_20_group"] = out["is_wide_prev1_region_outlier_20pct"].astype("Int8").astype("string")
    return out


def main() -> int:
    print("python", sys.version)
    print("tensorflow", tf.__version__)
    print("pandas", pd.__version__)
    print("project", PROJECT_DIR)
    print("run_mode", RUN_MODE, "max_epochs", MAX_EPOCHS, "batch_size", BATCH_SIZE)
    ensure_outlier_sidecar()

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
    outlier_df = pd.read_csv(OUTLIER_PATH, dtype={"transaction_id": "string", "sgg_lag1_source_ym": "string"})
    assert exact_df["transaction_id"].is_unique
    assert outlier_df["transaction_id"].is_unique
    assert len(exact_df) == len(raw_df), (len(exact_df), len(raw_df))
    assert len(outlier_df) == len(raw_df), (len(outlier_df), len(raw_df))

    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(outlier_df.drop(columns=["sgg_lag1_source_ym"]), on="transaction_id", how="left", validate="one_to_one")
    assert int(model_df["prev2_missing"].isna().sum()) == 0
    assert int(model_df["exact_prev1_missing"].isna().sum()) == 0
    assert int(model_df["outlier_signal_score"].isna().sum()) == 0
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df = add_e10_features(model_df)
    splits = e09.apply_smoke_sampling(e09.split_frames(model_df))
    coverage_df = pd.DataFrame(
        [
            {
                "split": s,
                "rows": len(splits[s]),
                "wide_prev_jump_20pct_rate": float(splits[s]["is_wide_prev_jump_20pct"].mean()),
                "exact_prev_jump_20pct_rate": float(splits[s]["is_exact_prev_jump_20pct"].mean()),
                "exact_wide_gap_10pct_rate": float(splits[s]["is_exact_wide_prev1_gap_10pct"].mean()),
                "wide_region_outlier_20pct_rate": float(splits[s]["is_wide_prev1_region_outlier_20pct"].mean()),
                "sgg_prior_missing_rate": float(splits[s]["sgg_prior_missing"].mean()),
            }
            for s in SPLIT_ORDER
        ]
    )
    print(coverage_df)

    metric_frames = []
    group_frames = []
    for config in EXPERIMENTS:
        metrics, groups = e09.train_and_predict(config, splits)
        metric_frames.append(metrics)
        group_frames.append(groups)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    group_metrics_df = pd.concat(group_frames, ignore_index=True)

    reference = metrics_df.loc[metrics_df["experiment_name"] == "F18_reference_recheck", ["split", "log_mae"]].rename(columns={"log_mae": "reference_f18_recheck_log_mae"})
    metrics_df = metrics_df.merge(reference, on="split", how="left")
    metrics_df["delta_vs_f18_recheck"] = metrics_df["log_mae"] - metrics_df["reference_f18_recheck_log_mae"]
    metrics_df.to_csv(METRICS_PATH, index=False)
    group_metrics_df.to_csv(GROUP_METRICS_PATH, index=False)

    eval_metrics = metrics_df.loc[metrics_df["split"].isin(EVAL_SPLITS)].copy()
    f18_recent_p99 = eval_metrics.loc[(eval_metrics["experiment_name"] == "F18_reference_recheck") & (eval_metrics["split"] == "recent_holdout"), "abs_pct_error_p99"].iloc[0]
    judgement_rows = []
    for name in [e["experiment_name"] for e in EXPERIMENTS if e["experiment_name"] != "F18_reference_recheck"]:
        cand = eval_metrics.loc[eval_metrics["experiment_name"] == name].set_index("split")
        recent_delta = float(cand.loc["recent_holdout", "delta_vs_f18_recheck"])
        test_delta = float(cand.loc["test", "delta_vs_f18_recheck"])
        valid_delta = float(cand.loc["valid", "delta_vs_f18_recheck"])
        p99_delta = float(cand.loc["recent_holdout", "abs_pct_error_p99"] - f18_recent_p99)
        gt20_delta = float(
            cand.loc["recent_holdout", "error_gt_20pct_rate"]
            - eval_metrics.loc[(eval_metrics["experiment_name"] == "F18_reference_recheck") & (eval_metrics["split"] == "recent_holdout"), "error_gt_20pct_rate"].iloc[0]
        )
        success = recent_delta < -1e-9 and valid_delta <= 0.0005 and test_delta <= 0.0005 and p99_delta <= 0.01 and gt20_delta <= 0.002
        severe_degrade = recent_delta > 0.002 or test_delta > 0.002 or p99_delta > 0.03 or gt20_delta > 0.005
        judgement_rows.append(
            {
                "experiment_name": name,
                "valid_delta_vs_f18_recheck": valid_delta,
                "test_delta_vs_f18_recheck": test_delta,
                "recent_delta_vs_f18_recheck": recent_delta,
                "recent_p99_abs_pct_delta": p99_delta,
                "recent_gt20_rate_delta": gt20_delta,
                "judgement": "성공" if success else ("실패" if severe_degrade else "보류"),
            }
        )
    judgement_df = pd.DataFrame(judgement_rows)
    overall = "성공" if (judgement_df["judgement"] == "성공").any() else ("실패" if (judgement_df["judgement"] == "실패").all() else "보류")
    pivot = eval_metrics.pivot(index="experiment_name", columns="split", values="log_mae").reset_index()
    tail = eval_metrics[["experiment_name", "split", "abs_pct_error_p95", "abs_pct_error_p99", "error_gt_10pct_rate", "error_gt_20pct_rate", "delta_vs_f18_recheck"]]
    focused_groups = group_metrics_df.loc[
        (group_metrics_df["split"] == "recent_holdout")
        & (
            ((group_metrics_df["group_type"].isin(["wide_prev_jump_20_group", "exact_prev_jump_20_group", "exact_wide_gap_10_group", "wide_region_outlier_20_group"])) & (group_metrics_df["group_value"] == "1"))
            | ((group_metrics_df["group_type"].isin(["exact_prev1_gap_bucket_plus", "prev1_gap_bucket_plus", "prev2_gap_bucket_plus"])) & (group_metrics_df["group_value"].isin(["366-730", "731+"])))
        )
    ].copy()
    lines = [
        "# E10 outlier-signal feature 실험 요약",
        "",
        "## 1. 결론",
        f"- 결론: `{overall}`",
        "- 비교 기준은 동일 실행의 `F18_reference_recheck`입니다.",
        "- 채택 기준: `recent_holdout log_mae` 개선, valid/test 악화 제한, recent p99 및 gt20 tail 악화 제한.",
        "",
        md_table(judgement_df),
        "",
        "## 2. 실행 설정",
        f"- run_mode: `{RUN_MODE}`",
        f"- max_epochs: `{MAX_EPOCHS}`",
        f"- batch_size: `{BATCH_SIZE}`",
        "- split: `train<=2023`, `valid=2024`, `test=2025`, `recent_holdout>=2026`",
        "- Policy B: `is_cancelled == 0`, `trade_type in [중개거래, unknown]`",
        "- leakage guard: 현재 거래가격 기반 outlier flag는 사용하지 않습니다.",
        "",
        "## 3. Split row 수와 outlier-signal coverage",
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
        f"- `{OUTLIER_PATH}`",
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
