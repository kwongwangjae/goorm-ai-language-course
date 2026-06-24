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
E09_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_e09_exact_prev_experiments.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "e11_f18_oof_predictions.csv"
FEATURE_PATH = OUTPUT_DIR / "e11_region_residual_features.csv"
CONFIDENCE_REPORT_PATH = OUTPUT_DIR / "e11_region_confidence_report.csv"
POLICY_PATH = OUTPUT_DIR / "e11_prediction_interval_policy.md"
QUALITY_REPORT_PATH = OUTPUT_DIR / "e11_region_residual_feature_quality_report.md"

RUN_MODE = os.environ.get("E11_RUN_MODE", "full").strip().lower()
REBUILD_OOF = os.environ.get("E11_REBUILD_OOF", "0") == "1"
REBUILD_FEATURES = os.environ.get("E11_REBUILD_RESIDUAL_FEATURES", "0") == "1"
RANDOM_STATE = 42
BATCH_SIZE = int(os.environ.get("E11_BATCH_SIZE", "8192"))
MAX_EPOCHS = int(os.environ.get("E11_MAX_EPOCHS", "2" if RUN_MODE == "smoke" else "30"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("E11_EARLY_STOPPING_PATIENCE", "2" if RUN_MODE == "smoke" else "4"))
TRAIN_VERBOSE = int(os.environ.get("E11_TRAIN_VERBOSE", "2"))
SMOKE_LIMITS = {"train": 200_000, "valid": 50_000, "target": 50_000}
CONFIDENCE_SOURCE_UNTIL_YM = os.environ.get("E11_CONFIDENCE_SOURCE_UNTIL_YM", "2025-12")

assert RUN_MODE in {"smoke", "full"}, RUN_MODE
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

LEVELS = [
    ("complex", "complex_id", 50.0),
    ("legal_dong", "legal_dong_code", 100.0),
    ("sgg", "sgg_code", 300.0),
    ("sido", "sido_code", 1000.0),
]


def md_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float", "float32", "float64"]).columns:
        x[col] = x[col].map(lambda v: format(v, floatfmt) if pd.notna(v) else "")
    x = x.astype("string").fillna("")
    lines = ["| " + " | ".join(x.columns) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in x.values.tolist()]
    return "\n".join(lines)


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


F18_CONFIG = {
    "experiment_name": "F18_exact_area_prev_additive",
    "numeric_features": list(e09.EXACT_ADDITIVE_FEATURES),
    "base_log_feature": "log_complex_prev_price_per_m2",
    "embedding_features": e09.BASE_EMBEDDING_FEATURES,
    "embedding_dims": e09.EMBEDDING_DIMS,
    "learning_rate": 0.001,
    "dense_units": [128, 64],
    "seed_offset": 18,
}


def ym_to_index_series(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace("-", "", regex=False)
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce")
    month = pd.to_numeric(text.str.slice(4, 6), errors="coerce")
    return (year * 12 + month - 1).astype("Int64")


def ym_to_index(value: str) -> int:
    text = value.replace("-", "")
    return int(text[:4]) * 12 + int(text[4:6]) - 1


def index_to_ym(value: int | float | pd.NA) -> str:
    if pd.isna(value):
        return ""
    idx = int(value)
    year, month0 = divmod(idx, 12)
    return f"{year:04d}-{month0 + 1:02d}"


def source_until_index_for_deal(deal_idx: pd.Series) -> pd.Series:
    prev_idx = deal_idx.astype("Int64") - 1
    valid_cap = ym_to_index("2023-12")
    test_cap = ym_to_index("2024-12")
    recent_cap = ym_to_index("2025-12")
    valid_start = ym_to_index("2024-01")
    test_start = ym_to_index("2025-01")
    recent_start = ym_to_index("2026-01")
    out = prev_idx.copy()
    out = out.mask(deal_idx.ge(valid_start) & deal_idx.lt(test_start), np.minimum(prev_idx, valid_cap))
    out = out.mask(deal_idx.ge(test_start) & deal_idx.lt(recent_start), np.minimum(prev_idx, test_cap))
    out = out.mask(deal_idx.ge(recent_start), np.minimum(prev_idx, recent_cap))
    return out.astype("Int64")


def sample_frame(frame: pd.DataFrame, limit: int, sort_cols: list[str] | None = None) -> pd.DataFrame:
    if RUN_MODE != "smoke" or len(frame) <= limit:
        return frame.copy()
    sampled = frame.sample(n=limit, random_state=RANDOM_STATE)
    return sampled.sort_values(sort_cols or ["deal_date", "transaction_id"]).copy()


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
    if not raw_df["transaction_id"].is_unique:
        raise RuntimeError("transactions.csv transaction_id is not unique")
    if not prev2_df["transaction_id"].is_unique:
        raise RuntimeError("e07 prev2 transaction_id is not unique")
    if not exact_df["transaction_id"].is_unique:
        raise RuntimeError("e09 exact prev transaction_id is not unique")
    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    if int(model_df["prev2_missing"].isna().sum()) != 0:
        raise RuntimeError("prev2 sidecar join missing")
    if int(model_df["exact_prev1_missing"].isna().sum()) != 0:
        raise RuntimeError("exact prev sidecar join missing")
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df = e09.add_model_features(model_df)
    model_df["sido_code"] = model_df["sgg_code"].fillna("missing").astype("string").str.slice(0, 2)
    model_df["year"] = model_df["deal_date"].dt.year.astype("int16")
    return model_df.sort_values(["deal_date", "transaction_id"]).reset_index(drop=True)


def train_fold_predict(train_core: pd.DataFrame, valid_frame: pd.DataFrame, target_frame: pd.DataFrame, train_until_year: int) -> np.ndarray:
    tf.keras.backend.clear_session()
    medians = e09.numeric_medians_for(F18_CONFIG, {"train": train_core})
    train_inputs, normalizer, lookups = e09.build_preprocessors(F18_CONFIG, train_core, medians)
    model = e09.build_model(F18_CONFIG, normalizer, lookups)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    print(
        "fold",
        train_until_year,
        "train_rows",
        len(train_core),
        "valid_rows",
        len(valid_frame),
        "target_rows",
        len(target_frame),
    )
    model.fit(
        train_inputs,
        e09.y_for(train_core, medians, F18_CONFIG),
        validation_data=(e09.make_inputs(valid_frame, F18_CONFIG, medians), e09.y_for(valid_frame, medians, F18_CONFIG)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=TRAIN_VERBOSE,
    )
    raw_pred = model.predict(e09.make_inputs(target_frame, F18_CONFIG, medians), batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    pred_log = e09.final_log_pred(target_frame, raw_pred, medians, F18_CONFIG)
    if not np.isfinite(pred_log).all():
        raise RuntimeError(f"non-finite prediction in train_until={train_until_year}")
    return pred_log


def split_label_for_year(year: int) -> str:
    if year <= 2023:
        return "train_oof"
    if year == 2024:
        return "valid"
    if year == 2025:
        return "test"
    return "recent_holdout"


def build_oof_predictions() -> pd.DataFrame:
    if OOF_PREDICTIONS_PATH.exists() and not REBUILD_OOF:
        print("reuse_oof", OOF_PREDICTIONS_PATH)
        return pd.read_csv(
            OOF_PREDICTIONS_PATH,
            dtype={
                "transaction_id": "string",
                "split": "string",
                "deal_ym": "string",
                "complex_id": "string",
                "legal_dong_code": "string",
                "sgg_code": "string",
                "sido_code": "string",
                "model_train_until_ym": "string",
            },
            parse_dates=["deal_date"],
        )

    model_df = load_model_frame()
    fold_specs = [
        (2021, 2020, model_df["year"].eq(2021)),
        (2022, 2021, model_df["year"].eq(2022)),
        (2023, 2022, model_df["year"].eq(2023)),
        (2024, 2023, model_df["year"].eq(2024)),
        (2025, 2024, model_df["year"].eq(2025)),
        (2026, 2025, model_df["year"].ge(2026)),
    ]
    frames: list[pd.DataFrame] = []
    for target_year, train_until_year, target_mask in fold_specs:
        train_core = model_df.loc[model_df["year"] < train_until_year].copy()
        valid_frame = model_df.loc[model_df["year"].eq(train_until_year)].copy()
        target_frame = model_df.loc[target_mask].copy()
        if len(target_frame) == 0 or len(train_core) == 0 or len(valid_frame) == 0:
            print("skip_fold", target_year, "train", len(train_core), "valid", len(valid_frame), "target", len(target_frame))
            continue
        train_core = sample_frame(train_core, SMOKE_LIMITS["train"])
        valid_frame = sample_frame(valid_frame, SMOKE_LIMITS["valid"])
        target_frame = sample_frame(target_frame, SMOKE_LIMITS["target"])
        start = time.perf_counter()
        pred_log = train_fold_predict(train_core, valid_frame, target_frame, train_until_year)
        actual_log = target_frame["target"].to_numpy(dtype="float64")
        actual_ppm = target_frame["price_per_m2"].to_numpy(dtype="float64")
        pred_ppm = np.exp(pred_log)
        residual_log = actual_log - pred_log
        signed_pct = (pred_ppm - actual_ppm) / actual_ppm
        out = target_frame[
            [
                "transaction_id",
                "deal_date",
                "deal_ym",
                "complex_id",
                "legal_dong_code",
                "sgg_code",
                "sido_code",
            ]
        ].copy()
        out["split"] = split_label_for_year(target_year)
        out["actual_log_price_per_m2"] = actual_log
        out["pred_log_price_per_m2"] = pred_log
        out["actual_price_per_m2"] = actual_ppm
        out["pred_price_per_m2"] = pred_ppm
        out["residual_log"] = residual_log
        out["abs_log_error"] = np.abs(residual_log)
        out["signed_pct_error"] = signed_pct
        out["abs_pct_error"] = np.abs(signed_pct)
        out["model_train_until_ym"] = f"{train_until_year:04d}-12"
        ordered = [
            "transaction_id",
            "split",
            "deal_date",
            "deal_ym",
            "complex_id",
            "legal_dong_code",
            "sgg_code",
            "sido_code",
            "actual_log_price_per_m2",
            "pred_log_price_per_m2",
            "actual_price_per_m2",
            "pred_price_per_m2",
            "residual_log",
            "abs_log_error",
            "signed_pct_error",
            "abs_pct_error",
            "model_train_until_ym",
        ]
        frames.append(out[ordered])
        print("fold_done", target_year, "seconds", round(time.perf_counter() - start, 2))
    if not frames:
        raise RuntimeError("no OOF prediction folds were built")
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(OOF_PREDICTIONS_PATH, index=False)
    print("oof_predictions", OOF_PREDICTIONS_PATH, "rows", len(oof))
    return oof


def load_base_identity_frame() -> pd.DataFrame:
    usecols = [
        "transaction_id",
        "complex_id",
        "normalized_complex_name",
        "legal_dong_code",
        "sgg_code",
        "deal_ym",
    ]
    dtypes = {
        "transaction_id": "string",
        "complex_id": "string",
        "normalized_complex_name": "string",
        "legal_dong_code": "string",
        "sgg_code": "string",
        "deal_ym": "string",
    }
    base = pd.read_csv(DATA_PATH, usecols=usecols, dtype=dtypes)
    if not base["transaction_id"].is_unique:
        raise RuntimeError("transactions.csv transaction_id is not unique")
    base["sido_code"] = base["sgg_code"].fillna("missing").astype("string").str.slice(0, 2)
    base["deal_ym_idx"] = ym_to_index_series(base["deal_ym"])
    base["resid_source_until_idx"] = source_until_index_for_deal(base["deal_ym_idx"])
    base["resid_source_until_ym"] = base["resid_source_until_idx"].map(index_to_ym).astype("string")
    for _, col, _ in LEVELS:
        base[col] = base[col].fillna("missing").astype("string")
    return base


def aggregate_monthly_residuals(oof: pd.DataFrame, level_col: str) -> dict[str, dict[str, np.ndarray]]:
    work = oof[[level_col, "deal_ym", "residual_log", "abs_log_error", "abs_pct_error"]].copy()
    work[level_col] = work[level_col].fillna("missing").astype("string")
    work["deal_ym_idx"] = ym_to_index_series(work["deal_ym"])
    work["gt10"] = work["abs_pct_error"].astype("float64").gt(0.10).astype("int64")
    work["gt20"] = work["abs_pct_error"].astype("float64").gt(0.20).astype("int64")
    monthly = (
        work.groupby([level_col, "deal_ym_idx"], dropna=False)
        .agg(
            count=("residual_log", "size"),
            sum_residual_log=("residual_log", "sum"),
            sum_abs_log_error=("abs_log_error", "sum"),
            sum_abs_pct_error=("abs_pct_error", "sum"),
            sum_gt10=("gt10", "sum"),
            sum_gt20=("gt20", "sum"),
        )
        .reset_index()
        .sort_values([level_col, "deal_ym_idx"])
    )
    out: dict[str, dict[str, np.ndarray]] = {}
    for code, group in monthly.groupby(level_col, sort=False, observed=True):
        g = group.sort_values("deal_ym_idx")
        out[str(code)] = {
            "month": g["deal_ym_idx"].to_numpy(dtype="int64"),
            "count": g["count"].cumsum().to_numpy(dtype="float64"),
            "sum_residual_log": g["sum_residual_log"].cumsum().to_numpy(dtype="float64"),
            "sum_abs_log_error": g["sum_abs_log_error"].cumsum().to_numpy(dtype="float64"),
            "sum_abs_pct_error": g["sum_abs_pct_error"].cumsum().to_numpy(dtype="float64"),
            "sum_gt10": g["sum_gt10"].cumsum().to_numpy(dtype="float64"),
            "sum_gt20": g["sum_gt20"].cumsum().to_numpy(dtype="float64"),
        }
    return out


def assign_level_features(base: pd.DataFrame, oof: pd.DataFrame, prefix: str, level_col: str, shrink_k: float) -> pd.DataFrame:
    groups = aggregate_monthly_residuals(oof, level_col)
    count = np.zeros(len(base), dtype="float64")
    sum_resid = np.zeros(len(base), dtype="float64")
    sum_abs_log = np.zeros(len(base), dtype="float64")
    sum_abs_pct = np.zeros(len(base), dtype="float64")
    sum_gt10 = np.zeros(len(base), dtype="float64")
    sum_gt20 = np.zeros(len(base), dtype="float64")

    source_idx = base["resid_source_until_idx"].to_numpy(dtype="int64", na_value=-10**9)
    codes = base[level_col].fillna("missing").astype("string")
    for code, index in codes.groupby(codes, sort=False).groups.items():
        stats = groups.get(str(code))
        if stats is None:
            continue
        loc = np.fromiter(index, dtype="int64")
        pos = np.searchsorted(stats["month"], source_idx[loc], side="right") - 1
        valid = pos >= 0
        if not valid.any():
            continue
        target_loc = loc[valid]
        stat_pos = pos[valid]
        count[target_loc] = stats["count"][stat_pos]
        sum_resid[target_loc] = stats["sum_residual_log"][stat_pos]
        sum_abs_log[target_loc] = stats["sum_abs_log_error"][stat_pos]
        sum_abs_pct[target_loc] = stats["sum_abs_pct_error"][stat_pos]
        sum_gt10[target_loc] = stats["sum_gt10"][stat_pos]
        sum_gt20[target_loc] = stats["sum_gt20"][stat_pos]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_resid = np.divide(sum_resid, count, out=np.zeros_like(sum_resid), where=count > 0)
        mean_abs_log = np.divide(sum_abs_log, count, out=np.zeros_like(sum_abs_log), where=count > 0)
        mean_abs_pct = np.divide(sum_abs_pct, count, out=np.zeros_like(sum_abs_pct), where=count > 0)
        gt10_rate = np.divide(sum_gt10, count, out=np.zeros_like(sum_gt10), where=count > 0)
        gt20_rate = np.divide(sum_gt20, count, out=np.zeros_like(sum_gt20), where=count > 0)
    confidence = count / (count + shrink_k)
    return pd.DataFrame(
        {
            f"{prefix}_resid_source_count": count.astype("int64"),
            f"{prefix}_resid_bias_log_shrunk": (mean_resid * confidence).astype("float32"),
            f"{prefix}_resid_abs_log_mean": mean_abs_log.astype("float32"),
            f"{prefix}_resid_abs_pct_mean": mean_abs_pct.astype("float32"),
            f"{prefix}_resid_error_gt_10_rate": gt10_rate.astype("float32"),
            f"{prefix}_resid_error_gt_20_rate": gt20_rate.astype("float32"),
            f"{prefix}_resid_count_log1p": np.log1p(count).astype("float32"),
            f"{prefix}_resid_confidence": confidence.astype("float32"),
        },
        index=base.index,
    )


def build_residual_sidecar(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if FEATURE_PATH.exists() and CONFIDENCE_REPORT_PATH.exists() and not REBUILD_FEATURES and not REBUILD_OOF:
        print("reuse_residual_features", FEATURE_PATH)
        sidecar = pd.read_csv(FEATURE_PATH, dtype={"transaction_id": "string", "resid_source_until_ym": "string", "resid_risk_tier": "string"})
        report = pd.read_csv(CONFIDENCE_REPORT_PATH, dtype={"level": "string", "code": "string", "source_until_ym": "string", "confidence_tier": "string"})
        return sidecar, report

    base = load_base_identity_frame()
    oof = oof.copy()
    for _, col, _ in LEVELS:
        oof[col] = oof[col].fillna("missing").astype("string")

    sidecar = pd.DataFrame(
        {
            "transaction_id": base["transaction_id"],
            "resid_source_until_ym": base["resid_source_until_ym"],
        }
    )
    level_feature_frames = {}
    for prefix, level_col, shrink_k in LEVELS:
        features = assign_level_features(base, oof, prefix, level_col, shrink_k)
        level_feature_frames[prefix] = features
        sidecar = pd.concat([sidecar, features], axis=1)

    confidence_cols = [f"{prefix}_resid_confidence" for prefix, _, _ in LEVELS]
    bias_cols = [f"{prefix}_resid_bias_log_shrunk" for prefix, _, _ in LEVELS]
    conf = sidecar[confidence_cols].to_numpy(dtype="float64")
    bias = sidecar[bias_cols].to_numpy(dtype="float64")
    weights = conf.sum(axis=1)
    sidecar["blended_resid_bias_log"] = np.divide((conf * bias).sum(axis=1), weights, out=np.zeros(len(sidecar), dtype="float64"), where=weights > 0).astype("float32")

    count_cols = [f"{prefix}_resid_source_count" for prefix, _, _ in LEVELS]
    abs_pct_cols = [f"{prefix}_resid_abs_pct_mean" for prefix, _, _ in LEVELS]
    gt20_cols = [f"{prefix}_resid_error_gt_20_rate" for prefix, _, _ in LEVELS]
    level_names = np.array([prefix for prefix, _, _ in LEVELS], dtype=object)
    counts = sidecar[count_cols].to_numpy(dtype="float64")
    confs = sidecar[confidence_cols].to_numpy(dtype="float64")
    abs_pct = sidecar[abs_pct_cols].to_numpy(dtype="float64")
    gt20 = sidecar[gt20_cols].to_numpy(dtype="float64")
    best_idx = np.argmax(confs, axis=1)
    row_idx = np.arange(len(sidecar))
    best_count = counts[row_idx, best_idx]
    best_conf = confs[row_idx, best_idx]
    best_abs_pct = abs_pct[row_idx, best_idx]
    best_gt20 = gt20[row_idx, best_idx]
    sidecar["best_resid_level"] = pd.Series(level_names[best_idx], index=sidecar.index).where(best_count > 0, "global").astype("string")
    sidecar["best_resid_source_count"] = best_count.astype("int64")
    sidecar["best_resid_abs_pct_mean"] = best_abs_pct.astype("float32")
    sidecar["resid_expected_abs_pct_error"] = best_abs_pct.astype("float32")
    risk = np.full(len(sidecar), "high", dtype=object)
    risk[best_count <= 0] = "unknown"
    risk[(best_count > 0) & (best_abs_pct <= 0.10) & (best_gt20 <= 0.08)] = "medium"
    risk[(best_count > 0) & (best_conf >= 0.25) & (best_abs_pct <= 0.06) & (best_gt20 <= 0.03)] = "low"
    sidecar["resid_risk_tier"] = pd.Series(risk, index=sidecar.index).astype("string")

    numeric_cols = sidecar.select_dtypes(include=["number"]).columns
    sidecar[numeric_cols] = sidecar[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    sidecar.to_csv(FEATURE_PATH, index=False)
    confidence_oof = filter_confidence_source(oof)
    report = build_confidence_report(confidence_oof, base)
    report.to_csv(CONFIDENCE_REPORT_PATH, index=False)
    write_prediction_interval_policy(report)
    write_quality_report(base, sidecar, oof)
    return sidecar, report


def filter_confidence_source(oof: pd.DataFrame) -> pd.DataFrame:
    cap_idx = ym_to_index(CONFIDENCE_SOURCE_UNTIL_YM)
    deal_idx = ym_to_index_series(oof["deal_ym"])
    filtered = oof.loc[deal_idx.le(cap_idx)].copy()
    if filtered.empty:
        raise RuntimeError(f"confidence source filter removed all residuals: <= {CONFIDENCE_SOURCE_UNTIL_YM}")
    return filtered


def confidence_tier(rows: int, p90: float) -> str:
    if rows >= 300 and p90 <= 0.20:
        return "high"
    if rows >= 50 and p90 <= 0.30:
        return "medium"
    if rows > 0:
        return "low"
    return "no_history"


def summarize_report_group(frame: pd.DataFrame, level: str, code_col: str, names: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for code, group in frame.groupby(code_col, dropna=False, observed=True):
        abs_pct = group["abs_pct_error"].astype("float64")
        signed = group["signed_pct_error"].astype("float64")
        row_count = int(len(group))
        p90 = float(abs_pct.quantile(0.90))
        rows.append(
            {
                "level": level,
                "code": str(code),
                "name": (names or {}).get(str(code), str(code)),
                "source_until_ym": str(group["deal_ym"].max()),
                "rows": row_count,
                "mean_abs_pct_error": float(abs_pct.mean()),
                "median_abs_pct_error": float(abs_pct.median()),
                "p80_abs_pct_error": float(abs_pct.quantile(0.80)),
                "p90_abs_pct_error": p90,
                "p95_abs_pct_error": float(abs_pct.quantile(0.95)),
                "error_gt_10pct_rate": float(abs_pct.gt(0.10).mean()),
                "error_gt_20pct_rate": float(abs_pct.gt(0.20).mean()),
                "mean_signed_pct_error": float(signed.mean()),
                "confidence_tier": confidence_tier(row_count, p90),
            }
        )
    return pd.DataFrame(rows)


def build_confidence_report(oof: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    complex_names = (
        base[["complex_id", "normalized_complex_name"]]
        .dropna()
        .drop_duplicates("complex_id")
        .assign(complex_id=lambda x: x["complex_id"].astype(str), normalized_complex_name=lambda x: x["normalized_complex_name"].astype(str))
        .set_index("complex_id")["normalized_complex_name"]
        .to_dict()
    )
    reports = [
        summarize_report_group(oof, "complex", "complex_id", complex_names),
        summarize_report_group(oof, "legal_dong", "legal_dong_code"),
        summarize_report_group(oof, "sgg", "sgg_code"),
        summarize_report_group(oof, "sido", "sido_code"),
    ]
    global_abs = oof["abs_pct_error"].astype("float64")
    global_signed = oof["signed_pct_error"].astype("float64")
    global_p90 = float(global_abs.quantile(0.90))
    reports.append(
        pd.DataFrame(
            [
                {
                    "level": "global",
                    "code": "global",
                    "name": "global",
                    "source_until_ym": str(oof["deal_ym"].max()),
                    "rows": int(len(oof)),
                    "mean_abs_pct_error": float(global_abs.mean()),
                    "median_abs_pct_error": float(global_abs.median()),
                    "p80_abs_pct_error": float(global_abs.quantile(0.80)),
                    "p90_abs_pct_error": global_p90,
                    "p95_abs_pct_error": float(global_abs.quantile(0.95)),
                    "error_gt_10pct_rate": float(global_abs.gt(0.10).mean()),
                    "error_gt_20pct_rate": float(global_abs.gt(0.20).mean()),
                    "mean_signed_pct_error": float(global_signed.mean()),
                    "confidence_tier": confidence_tier(int(len(oof)), global_p90),
                }
            ]
        )
    )
    return pd.concat(reports, ignore_index=True).sort_values(["level", "rows"], ascending=[True, False])


def write_prediction_interval_policy(report: pd.DataFrame) -> None:
    global_row = report.loc[report["level"].eq("global")].iloc[0]
    tier_policy = (
        report.groupby("confidence_tier", dropna=False)
        .agg(
            groups=("code", "size"),
            median_p80=("p80_abs_pct_error", "median"),
            median_p90=("p90_abs_pct_error", "median"),
            median_p95=("p95_abs_pct_error", "median"),
        )
        .reset_index()
    )
    lines = [
        "# E11 prediction interval policy",
        "",
        "## 1. 결론",
        "- 가격 예측값 자체와 별개로, residual OOF 이력에서 지역/단지별 예상 오차 범위를 제공합니다.",
        "- 기본 표기는 `예상 오차 범위: ±p90`이며, 보수적 안내가 필요하면 `±p95`를 사용합니다.",
        "- residual source는 현재 거래월보다 이전 월까지만 사용합니다.",
        f"- confidence report 기본 source 상한은 `{CONFIDENCE_SOURCE_UNTIL_YM}`입니다.",
        "",
        "## 2. Global fallback",
        f"- source_until_ym: `{global_row['source_until_ym']}`",
        f"- rows: `{int(global_row['rows']):,}`",
        f"- p80_abs_pct_error: `{float(global_row['p80_abs_pct_error']):.4f}`",
        f"- p90_abs_pct_error: `{float(global_row['p90_abs_pct_error']):.4f}`",
        f"- p95_abs_pct_error: `{float(global_row['p95_abs_pct_error']):.4f}`",
        "",
        "## 3. Tier policy",
        md_table(tier_policy),
        "",
        "## 4. 사용 규칙",
        "- `complex` 통계가 충분하면 complex 기준 p90을 우선 사용합니다.",
        "- complex 이력이 부족하면 `legal_dong -> sgg -> sido -> global` 순서로 fallback합니다.",
        "- `confidence_tier=low` 또는 `resid_risk_tier=high|unknown`이면 가격 대신 신뢰도 안내를 우선 노출합니다.",
        f"- confidence_report_csv: `{CONFIDENCE_REPORT_PATH}`",
    ]
    POLICY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(base: pd.DataFrame, sidecar: pd.DataFrame, oof: pd.DataFrame) -> None:
    merged = base[["transaction_id", "deal_ym", "deal_ym_idx"]].merge(sidecar, on="transaction_id", how="left", validate="one_to_one")
    join_missing = int(merged["resid_source_until_ym"].isna().sum())
    source_idx = ym_to_index_series(merged["resid_source_until_ym"])
    numeric_cols = sidecar.select_dtypes(include=["number"]).columns
    checks = {
        "sidecar_rows_match_transactions": len(sidecar) == len(base),
        "transaction_id_unique": bool(sidecar["transaction_id"].is_unique),
        "join_missing_zero": join_missing == 0,
        "resid_source_until_before_deal_ym": bool((source_idx < merged["deal_ym_idx"]).all()),
        "numeric_features_finite": bool(np.isfinite(sidecar[numeric_cols].to_numpy(dtype="float64")).all()),
        "oof_predictions_unique_transaction_id": bool(oof["transaction_id"].is_unique),
        "oof_residual_finite": bool(np.isfinite(oof[["residual_log", "abs_log_error", "abs_pct_error"]].to_numpy(dtype="float64")).all()),
    }
    grade = "Pass" if all(checks.values()) else "Fail"
    failed = [name for name, ok in checks.items() if not ok]
    coverage = pd.DataFrame(
        [
            {"metric": "transactions_rows", "value": len(base)},
            {"metric": "sidecar_rows", "value": len(sidecar)},
            {"metric": "oof_prediction_rows", "value": len(oof)},
            {"metric": "complex_has_source_rate", "value": float(sidecar["complex_resid_source_count"].gt(0).mean())},
            {"metric": "legal_dong_has_source_rate", "value": float(sidecar["legal_dong_resid_source_count"].gt(0).mean())},
            {"metric": "sgg_has_source_rate", "value": float(sidecar["sgg_resid_source_count"].gt(0).mean())},
            {"metric": "sido_has_source_rate", "value": float(sidecar["sido_resid_source_count"].gt(0).mean())},
            {"metric": "unknown_risk_rate", "value": float(sidecar["resid_risk_tier"].eq("unknown").mean())},
            {"metric": "high_risk_rate", "value": float(sidecar["resid_risk_tier"].eq("high").mean())},
        ]
    )
    risk_counts = sidecar["resid_risk_tier"].value_counts(dropna=False).rename_axis("resid_risk_tier").reset_index(name="rows")
    lines = [
        "# E11 region residual feature 품질 리포트",
        "",
        f"- 품질 등급: `{grade}`",
        f"- run_mode: `{RUN_MODE}`",
        f"- max_epochs: `{MAX_EPOCHS}`",
        f"- rows: `{len(sidecar):,}`",
        "",
        "## 지적사항",
        "- none" if not failed else "- 실패 checks: `" + "`, `".join(failed) + "`",
        "",
        "## 검증 근거 확인",
    ]
    for name, ok in checks.items():
        lines.append(f"- {name}: {'pass' if ok else 'fail'}")
    lines.extend(
        [
            "",
            "## Coverage",
            md_table(coverage),
            "",
            "## Risk tier",
            md_table(risk_counts),
            "",
            "## 검증 공백",
            "- smoke 모드는 residual source 예측 row를 샘플링하므로 coverage 수치는 full보다 낮을 수 있습니다.",
            "- sidecar는 `transactions.csv`를 수정하지 않고 `transaction_id` 기준으로만 조인합니다.",
            "- `resid_source_until_ym`은 항상 현재 row의 `deal_ym`보다 이전 월입니다.",
            f"- oof_predictions_csv: `{OOF_PREDICTIONS_PATH}`",
            f"- sidecar_csv: `{FEATURE_PATH}`",
            f"- confidence_report_csv: `{CONFIDENCE_REPORT_PATH}`",
        ]
    )
    QUALITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if grade != "Pass":
        raise RuntimeError(f"quality report failed: {failed}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("python", sys.version)
    print("tensorflow", tf.__version__)
    print("pandas", pd.__version__)
    print("project", PROJECT_DIR)
    print("run_mode", RUN_MODE, "max_epochs", MAX_EPOCHS, "batch_size", BATCH_SIZE)
    oof = build_oof_predictions()
    sidecar, report = build_residual_sidecar(oof)
    print("feature", FEATURE_PATH, "rows", len(sidecar))
    print("confidence_report", CONFIDENCE_REPORT_PATH, "rows", len(report))
    print("quality_report", QUALITY_REPORT_PATH)
    print("policy", POLICY_PATH)
    print("seconds", round(time.perf_counter() - start, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
