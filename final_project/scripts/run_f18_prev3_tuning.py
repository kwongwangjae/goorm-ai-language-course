#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(os.environ.get("FINAL_PROJECT_DIR", "/Users/gwongwangjae/goorm-ai-language-course/final_project"))
E10_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_e10_outlier_signal_experiments.py"
OUTPUT_DIR = REPO_ROOT / "outputs" / "f18_prev3_tuning"
PREV3_SIDECAR = OUTPUT_DIR / "f18_prev3_rolling_features.csv"
METRICS_CSV = OUTPUT_DIR / "f18_prev3_tuning_metrics.csv"
GROUP_METRICS_CSV = OUTPUT_DIR / "f18_prev3_tuning_group_metrics.csv"
SUMMARY_MD = OUTPUT_DIR / "f18_prev3_tuning_summary.md"
FINAL_DECISION_MD = OUTPUT_DIR / "f18_prev3_tuning_final_decision.md"
SUCCESS_PATH = OUTPUT_DIR / "_SUCCESS"

RANDOM_STATE = 42
EPOCH = int(os.environ.get("F18_PREV3_EPOCH", "30"))
EXACT_AREA_TOLERANCE_M2 = float(os.environ.get("F18_PREV3_EXACT_AREA_TOLERANCE_M2", "0.5"))
REBUILD_PREV3 = os.environ.get("F18_PREV3_REBUILD", "0") == "1"
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]

PREV3_FEATURES = [
    "log_complex_prev3_price_per_m2",
    "prev3_missing",
    "prev3_gap_months",
    "prev2_prev3_log_return",
    "prev2_prev3_gap_months",
    "complex_prev3_log_count",
    "complex_prev3_log_mean",
    "complex_prev3_log_median",
    "complex_prev3_log_std",
    "complex_prev3_log_spread",
    "log_exact_prev3_price_per_m2",
    "exact_prev3_missing",
    "exact_prev3_gap_months",
    "exact_prev2_prev3_log_return",
    "exact_prev2_prev3_gap_months",
    "exact_prev3_area_abs_diff",
    "exact_prev3_log_count",
    "exact_prev3_log_mean",
    "exact_prev3_log_median",
    "exact_prev3_log_std",
    "exact_prev3_log_spread",
]


@dataclass(frozen=True)
class Hist:
    area: float
    deal_date: pd.Timestamp
    ppm: float
    seq: int


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    valid_log_mae: float
    test_log_mae: float
    recent_log_mae: float
    recent_p95: float
    recent_p99: float
    recent_gt10: float
    recent_gt20: float
    valid_delta: float
    test_delta: float
    recent_delta: float
    p99_delta: float
    gt20_delta: float
    guardrail: str
    reasons: tuple[str, ...]


def load_e10_module() -> Any:
    spec = importlib.util.spec_from_file_location("e10_outlier_runner", E10_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E10_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_modules(e10: Any) -> None:
    batch_size = int(os.environ.get("F18_PREV3_BATCH_SIZE", os.environ.get("E10_BATCH_SIZE", "8192")))
    patience = int(os.environ.get("F18_PREV3_EARLY_STOPPING_PATIENCE", "4"))
    verbose = int(os.environ.get("F18_PREV3_TRAIN_VERBOSE", "2"))
    e10.RUN_MODE = "full"
    e10.BATCH_SIZE = batch_size
    e10.MAX_EPOCHS = EPOCH
    e10.EARLY_STOPPING_PATIENCE = patience
    e10.TRAIN_VERBOSE = verbose
    e10.e09.RUN_MODE = "full"
    e10.e09.BATCH_SIZE = batch_size
    e10.e09.MAX_EPOCHS = EPOCH
    e10.e09.EARLY_STOPPING_PATIENCE = patience
    e10.e09.TRAIN_VERBOSE = verbose


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def find_complex_prev3(hist: dict[str, list[Hist]], complex_id: str) -> Hist | None:
    items = hist.get(complex_id)
    if not items or len(items) < 3:
        return None
    return items[-3]


def find_exact_prev3(hist: dict[str, dict[float, list[Hist]]], complex_id: str, area_value: Any) -> Hist | None:
    area = to_float(area_value)
    if area is None or area <= 0 or not complex_id:
        return None
    buckets = hist.get(complex_id)
    if not buckets:
        return None
    candidates: list[Hist] = []
    lo = int(math.floor((area - EXACT_AREA_TOLERANCE_M2) * 10))
    hi = int(math.ceil((area + EXACT_AREA_TOLERANCE_M2) * 10))
    for bucket10 in range(lo, hi + 1):
        items = buckets.get(bucket10 / 10.0)
        if not items:
            continue
        found = 0
        for item in reversed(items):
            if abs(item.area - area) <= EXACT_AREA_TOLERANCE_M2:
                candidates.append(item)
                found += 1
                if found >= 3:
                    break
    candidates.sort(key=lambda item: (item.deal_date, item.seq), reverse=True)
    return candidates[2] if len(candidates) >= 3 else None


def build_prev3_sidecar(e10: Any) -> None:
    if PREV3_SIDECAR.exists() and not REBUILD_PREV3:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        e10.DATA_PATH,
        usecols=["transaction_id", "complex_id", "deal_date", "area_m2", "price_per_m2", "is_cancelled"],
        dtype={
            "transaction_id": "string",
            "complex_id": "string",
            "area_m2": "float32",
            "price_per_m2": "float32",
            "is_cancelled": "Int8",
        },
        parse_dates=["deal_date"],
    ).sort_values(["deal_date", "transaction_id"])
    if not raw["transaction_id"].is_unique:
        raise RuntimeError("transactions.csv transaction_id is not unique")

    complex_hist: dict[str, list[Hist]] = defaultdict(list)
    exact_hist: dict[str, dict[float, list[Hist]]] = defaultdict(lambda: defaultdict(list))
    rows: list[dict[str, Any]] = []
    seq = 0

    for deal_date, day in raw.groupby("deal_date", sort=True):
        day_rows = list(day.itertuples(index=False))
        for row in day_rows:
            complex_prev3 = find_complex_prev3(complex_hist, row.complex_id)
            exact_prev3 = find_exact_prev3(exact_hist, row.complex_id, row.area_m2)
            rows.append(
                {
                    "transaction_id": row.transaction_id,
                    "complex_prev3_price_per_m2": np.nan if complex_prev3 is None else complex_prev3.ppm,
                    "prev3_missing": 1 if complex_prev3 is None else 0,
                    "prev3_gap_days": np.nan if complex_prev3 is None else (deal_date - complex_prev3.deal_date).days,
                    "prev3_source_deal_date": "" if complex_prev3 is None else complex_prev3.deal_date.date().isoformat(),
                    "exact_prev3_price_per_m2": np.nan if exact_prev3 is None else exact_prev3.ppm,
                    "exact_prev3_missing": 1 if exact_prev3 is None else 0,
                    "exact_prev3_gap_days": np.nan if exact_prev3 is None else (deal_date - exact_prev3.deal_date).days,
                    "exact_prev3_source_deal_date": "" if exact_prev3 is None else exact_prev3.deal_date.date().isoformat(),
                    "exact_prev3_source_area_m2": np.nan if exact_prev3 is None else exact_prev3.area,
                }
            )
        for row in day_rows:
            area = to_float(row.area_m2)
            ppm = to_float(row.price_per_m2)
            if int(row.is_cancelled) != 0 or not row.complex_id or area is None or area <= 0 or ppm is None or ppm <= 0:
                continue
            seq += 1
            item = Hist(area=area, deal_date=deal_date, ppm=ppm, seq=seq)
            complex_hist[row.complex_id].append(item)
            exact_hist[row.complex_id][round(area, 1)].append(item)

    pd.DataFrame(rows).to_csv(PREV3_SIDECAR, index=False)


def load_model_frame(e10: Any) -> pd.DataFrame:
    e10.ensure_outlier_sidecar()
    build_prev3_sidecar(e10)
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
    raw_df = pd.read_csv(e10.DATA_PATH, usecols=base_usecols, dtype=base_dtypes, parse_dates=["deal_date"])
    prev2_df = pd.read_csv(e10.PREV2_PATH, dtype={"transaction_id": "string", "prev2_missing": "Int8"}, parse_dates=["prev2_source_deal_date"])
    exact_df = pd.read_csv(
        e10.EXACT_PREV_PATH,
        dtype={"transaction_id": "string", "exact_prev1_missing": "Int8", "exact_prev2_missing": "Int8"},
        parse_dates=["exact_prev1_source_deal_date", "exact_prev2_source_deal_date"],
    )
    outlier_df = pd.read_csv(e10.OUTLIER_PATH, dtype={"transaction_id": "string", "sgg_lag1_source_ym": "string"})
    prev3_df = pd.read_csv(
        PREV3_SIDECAR,
        dtype={"transaction_id": "string", "prev3_missing": "Int8", "exact_prev3_missing": "Int8"},
        parse_dates=["prev3_source_deal_date", "exact_prev3_source_deal_date"],
    )
    for label, frame in [("raw", raw_df), ("prev2", prev2_df), ("exact", exact_df), ("outlier", outlier_df), ("prev3", prev3_df)]:
        if not frame["transaction_id"].is_unique:
            raise RuntimeError(f"{label} transaction_id is not unique")

    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(outlier_df.drop(columns=["sgg_lag1_source_ym"]), on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(prev3_df, on="transaction_id", how="left", validate="one_to_one")
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df = e10.add_e10_features(model_df)
    return add_prev3_features(model_df).sort_values(["deal_date", "transaction_id"]).reset_index(drop=True)


def log_where_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return pd.Series(np.where(values > 0, np.log(values), np.nan), index=series.index, dtype="float64")


def rolling_stats(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    values = frame[columns].to_numpy(dtype="float64")
    valid = np.isfinite(values)
    count = valid.sum(axis=1).astype("float32")
    masked = np.where(valid, values, np.nan)
    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_count"] = count
    out[f"{prefix}_mean"] = np.nanmean(masked, axis=1).astype("float32")
    out[f"{prefix}_median"] = np.nanmedian(masked, axis=1).astype("float32")
    out[f"{prefix}_std"] = np.nanstd(masked, axis=1).astype("float32")
    out[f"{prefix}_spread"] = (np.nanmax(masked, axis=1) - np.nanmin(masked, axis=1)).astype("float32")
    for col in out.columns:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return out


def add_prev3_features(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    out["log_complex_prev3_price_per_m2"] = log_where_positive(out["complex_prev3_price_per_m2"]).astype("float32")
    out["log_exact_prev3_price_per_m2"] = log_where_positive(out["exact_prev3_price_per_m2"]).astype("float32")
    out["prev3_missing"] = out["prev3_missing"].fillna(1).astype("float32")
    out["exact_prev3_missing"] = out["exact_prev3_missing"].fillna(1).astype("float32")
    out["prev3_gap_months"] = out["prev3_gap_days"].astype("float64") / 30.4375
    out["exact_prev3_gap_months"] = out["exact_prev3_gap_days"].astype("float64") / 30.4375
    out["prev2_prev3_log_return"] = out["log_complex_prev2_price_per_m2"] - out["log_complex_prev3_price_per_m2"]
    out["prev2_prev3_gap_months"] = out["prev3_gap_months"] - out["prev2_gap_months"]
    out["exact_prev2_prev3_log_return"] = out["log_exact_prev2_price_per_m2"] - out["log_exact_prev3_price_per_m2"]
    out["exact_prev2_prev3_gap_months"] = out["exact_prev3_gap_months"] - out["exact_prev2_gap_months"]
    out["exact_prev3_area_abs_diff"] = (out["area_m2"].astype("float64") - out["exact_prev3_source_area_m2"].astype("float64")).abs()

    complex_stats = rolling_stats(
        out,
        ["log_complex_prev_price_per_m2", "log_complex_prev2_price_per_m2", "log_complex_prev3_price_per_m2"],
        "complex_prev3_log",
    )
    exact_stats = rolling_stats(
        out,
        ["log_exact_prev1_price_per_m2", "log_exact_prev2_price_per_m2", "log_exact_prev3_price_per_m2"],
        "exact_prev3_log",
    )
    for col in complex_stats.columns:
        out[col] = complex_stats[col]
    for col in exact_stats.columns:
        out[col] = exact_stats[col]
    for col in PREV3_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        if col.endswith("_missing"):
            out[col] = out[col].fillna(1)
        elif col.endswith("_count"):
            out[col] = out[col].fillna(0)
    return out


def experiment_configs(e10: Any) -> list[dict[str, Any]]:
    base_features = list(e10.F18_FEATURES)
    return [
        {
            "experiment_name": "F18_reference_huber_010",
            "numeric_features": base_features,
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": e10.e09.BASE_EMBEDDING_FEATURES,
            "embedding_dims": e10.e09.EMBEDDING_DIMS,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 183,
            "loss": "huber_010",
        },
        {
            "experiment_name": "F35_prev3_rolling_huber_010",
            "numeric_features": [*base_features, *PREV3_FEATURES],
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": e10.e09.BASE_EMBEDDING_FEATURES,
            "embedding_dims": e10.e09.EMBEDDING_DIMS,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 351,
            "loss": "huber_010",
        },
    ]


def score_candidates(metrics_df: pd.DataFrame, reference_name: str) -> list[CandidateScore]:
    ref = metrics_df.loc[metrics_df["experiment_name"] == reference_name].set_index("split")
    scores: list[CandidateScore] = []
    for candidate, group in metrics_df.groupby("experiment_name", sort=False):
        split_rows = group.set_index("split")
        if not all(split in split_rows.index for split in EVAL_SPLITS):
            continue
        valid = split_rows.loc["valid"]
        test = split_rows.loc["test"]
        recent = split_rows.loc["recent_holdout"]
        valid_delta = float(valid["log_mae"] - ref.loc["valid", "log_mae"])
        test_delta = float(test["log_mae"] - ref.loc["test", "log_mae"])
        recent_delta = float(recent["log_mae"] - ref.loc["recent_holdout", "log_mae"])
        p99_delta = float(recent["abs_pct_error_p99"] - ref.loc["recent_holdout", "abs_pct_error_p99"])
        gt20_delta = float(recent["error_gt_20pct_rate"] - ref.loc["recent_holdout", "error_gt_20pct_rate"])
        reasons: list[str] = []
        if valid_delta > 0.0005:
            reasons.append(f"valid log_mae delta {valid_delta:.6f} > 0.0005")
        if test_delta > 0.0005:
            reasons.append(f"test log_mae delta {test_delta:.6f} > 0.0005")
        if p99_delta > 0.003:
            reasons.append(f"recent p99 delta {p99_delta:.6f} > 0.003")
        if gt20_delta > 0.001:
            reasons.append(f"recent gt20 delta {gt20_delta:.6f} > 0.001")
        scores.append(
            CandidateScore(
                candidate=str(candidate),
                valid_log_mae=float(valid["log_mae"]),
                test_log_mae=float(test["log_mae"]),
                recent_log_mae=float(recent["log_mae"]),
                recent_p95=float(recent["abs_pct_error_p95"]),
                recent_p99=float(recent["abs_pct_error_p99"]),
                recent_gt10=float(recent["error_gt_10pct_rate"]),
                recent_gt20=float(recent["error_gt_20pct_rate"]),
                valid_delta=valid_delta,
                test_delta=test_delta,
                recent_delta=recent_delta,
                p99_delta=p99_delta,
                gt20_delta=gt20_delta,
                guardrail="baseline" if candidate == reference_name else ("fail" if reasons else "pass"),
                reasons=tuple(reasons),
            )
        )
    return scores


def fmt(value: float) -> str:
    return f"{value:.6f}"


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def write_outputs(metrics_df: pd.DataFrame, group_df: pd.DataFrame, scores: list[CandidateScore], reference_name: str) -> CandidateScore:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV, index=False)
    group_df.to_csv(GROUP_METRICS_CSV, index=False)
    eligible = [score for score in scores if score.guardrail in {"pass", "baseline"}]
    best = min(
        eligible,
        key=lambda score: (
            score.recent_log_mae,
            score.recent_gt10,
            score.recent_p99,
            0 if score.candidate == reference_name else 1,
        ),
    )
    ordered = sorted(scores, key=lambda score: (score.guardrail == "fail", score.recent_log_mae, score.recent_gt10, score.recent_p99))
    lines = [
        "# F18 Prev3 Rolling Tuning Summary",
        "",
        "## Best",
        "",
        f"- reference: `{reference_name}`",
        f"- best: `{best.candidate}`",
        f"- epoch: `{EPOCH}`",
        f"- guardrail: `{best.guardrail}`",
        "",
        "## Recent Holdout Ranking",
        "",
        "| rank | candidate | MAE | p95 | p99 | >10% | >20% | d_MAE | d_p99 | d_gt20 | guardrail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, score in enumerate(ordered, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    score.candidate,
                    fmt(score.recent_log_mae),
                    pct(score.recent_p95),
                    pct(score.recent_p99),
                    pct(score.recent_gt10),
                    pct(score.recent_gt20),
                    fmt(score.recent_delta),
                    pct(score.p99_delta),
                    pct(score.gt20_delta),
                    score.guardrail,
                ]
            )
            + " |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = [
        "# F18 Prev3 Rolling Tuning Decision",
        "",
        "## Decision",
        "",
        f"- final candidate: `{best.candidate}`",
        f"- epoch: `{EPOCH}`",
        f"- guardrail: `{best.guardrail}`",
        f"- recent_holdout log_mae: `{fmt(best.recent_log_mae)}`",
        f"- recent_holdout p95: `{pct(best.recent_p95)}`",
        f"- recent_holdout p99: `{pct(best.recent_p99)}`",
        f"- recent_holdout >10%: `{pct(best.recent_gt10)}`",
        f"- recent_holdout >20%: `{pct(best.recent_gt20)}`",
        "",
        "## Delta vs Reference",
        "",
        f"- log_mae: `{fmt(best.recent_delta)}`",
        f"- p99: `{pct(best.p99_delta)}`",
        f"- >20%: `{pct(best.gt20_delta)}`",
    ]
    FINAL_DECISION_MD.write_text("\n".join(decision) + "\n", encoding="utf-8")
    SUCCESS_PATH.write_text(f"completed_at_utc={pd.Timestamp.utcnow().isoformat()}\n", encoding="utf-8")
    return best


def main() -> int:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    e10 = load_e10_module()
    configure_modules(e10)
    print("project", PROJECT_DIR, flush=True)
    print("epoch", EPOCH, flush=True)
    print("output", OUTPUT_DIR, flush=True)
    model_df = load_model_frame(e10)
    splits = e10.e09.apply_smoke_sampling(e10.e09.split_frames(model_df))
    print(pd.DataFrame([{"split": name, "rows": len(frame)} for name, frame in splits.items()]), flush=True)

    metric_frames = []
    group_frames = []
    configs = experiment_configs(e10)
    for config in configs:
        print("\n===", config["experiment_name"], "===", flush=True)
        metrics, groups = e10.e09.train_and_predict(config, splits)
        metric_frames.append(metrics)
        group_frames.append(groups)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    group_df = pd.concat(group_frames, ignore_index=True)
    reference_name = "F18_reference_huber_010"
    scores = score_candidates(metrics_df, reference_name)
    best = write_outputs(metrics_df, group_df, scores, reference_name)
    print("best", best.candidate, fmt(best.recent_log_mae), best.guardrail, flush=True)
    print("seconds", round(time.perf_counter() - start, 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
