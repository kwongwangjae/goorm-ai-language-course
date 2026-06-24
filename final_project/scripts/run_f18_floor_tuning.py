#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(os.environ.get("FINAL_PROJECT_DIR", "/Users/gwongwangjae/goorm-ai-language-course/final_project"))
E10_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_e10_outlier_signal_experiments.py"
OUTPUT_DIR = REPO_ROOT / "outputs" / "f18_floor_tuning"
METRICS_CSV = OUTPUT_DIR / "f18_floor_tuning_metrics.csv"
GROUP_METRICS_CSV = OUTPUT_DIR / "f18_floor_tuning_group_metrics.csv"
SUMMARY_MD = OUTPUT_DIR / "f18_floor_tuning_summary.md"
FINAL_DECISION_MD = OUTPUT_DIR / "f18_floor_tuning_final_decision.md"
SUCCESS_PATH = OUTPUT_DIR / "_SUCCESS"

RANDOM_STATE = 42
EPOCH = int(os.environ.get("F18_FLOOR_EPOCH", "30"))
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]

FLOOR_FLAG_FEATURES = [
    "floor_abs",
    "floor_log1p_abs",
    "is_first_floor",
    "is_low_floor_1_3",
    "is_mid_floor_4_10",
    "is_high_floor_15p",
    "is_very_high_floor_25p",
]
FLOOR_INTERACTION_FEATURES = [
    "floor_area_interaction",
    "floor_age_interaction",
    "low_floor_age_interaction",
    "high_floor_area_interaction",
]


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
    batch_size = int(os.environ.get("F18_FLOOR_BATCH_SIZE", os.environ.get("E10_BATCH_SIZE", "8192")))
    patience = int(os.environ.get("F18_FLOOR_EARLY_STOPPING_PATIENCE", "4"))
    verbose = int(os.environ.get("F18_FLOOR_TRAIN_VERBOSE", "2"))
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
    e10.e09.EMBEDDING_DIMS["floor_bucket"] = 4


def load_model_frame(e10: Any) -> pd.DataFrame:
    e10.ensure_outlier_sidecar()
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
    for label, frame in [("raw", raw_df), ("prev2", prev2_df), ("exact", exact_df), ("outlier", outlier_df)]:
        if not frame["transaction_id"].is_unique:
            raise RuntimeError(f"{label} transaction_id is not unique")

    model_df = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    model_df = model_df.merge(outlier_df.drop(columns=["sgg_lag1_source_ym"]), on="transaction_id", how="left", validate="one_to_one")
    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df = e10.add_e10_features(model_df)
    return add_floor_features(model_df).sort_values(["deal_date", "transaction_id"]).reset_index(drop=True)


def add_floor_features(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    floor = out["floor"].astype("float32").fillna(0)
    floor_pos = floor.clip(lower=0)
    area = out["area_m2"].astype("float32").fillna(out["area_m2"].median())
    age = out["age_years"].astype("float32").fillna(out["age_years"].median())

    out["floor_abs"] = floor.abs().astype("float32")
    out["floor_log1p_abs"] = np.log1p(out["floor_abs"].astype("float64")).astype("float32")
    out["is_first_floor"] = (floor == 1).astype("float32")
    out["is_low_floor_1_3"] = ((floor >= 1) & (floor <= 3)).astype("float32")
    out["is_mid_floor_4_10"] = ((floor >= 4) & (floor <= 10)).astype("float32")
    out["is_high_floor_15p"] = (floor >= 15).astype("float32")
    out["is_very_high_floor_25p"] = (floor >= 25).astype("float32")
    out["floor_area_interaction"] = (floor * area).astype("float32")
    out["floor_age_interaction"] = (floor * age).astype("float32")
    out["low_floor_age_interaction"] = (out["is_low_floor_1_3"] * age).astype("float32")
    out["high_floor_area_interaction"] = (out["is_high_floor_15p"] * area).astype("float32")

    bucket = pd.Series("missing", index=out.index, dtype="string")
    bucket[floor < 0] = "basement"
    bucket[floor == 0] = "zero_or_missing"
    bucket[floor == 1] = "1"
    bucket[(floor >= 2) & (floor <= 3)] = "2-3"
    bucket[(floor >= 4) & (floor <= 6)] = "4-6"
    bucket[(floor >= 7) & (floor <= 10)] = "7-10"
    bucket[(floor >= 11) & (floor <= 15)] = "11-15"
    bucket[(floor >= 16) & (floor <= 20)] = "16-20"
    bucket[(floor >= 21) & (floor <= 30)] = "21-30"
    bucket[floor >= 31] = "31+"
    out["floor_bucket"] = bucket.fillna("missing").astype("string")
    out["floor_pos"] = floor_pos.astype("float32")
    return out


def experiment_configs(e10: Any) -> list[dict[str, Any]]:
    base_embedding_dims = dict(e10.e09.EMBEDDING_DIMS)
    floor_embedding_dims = {**base_embedding_dims, "floor_bucket": 4}
    base_embedding_features = list(e10.e09.BASE_EMBEDDING_FEATURES)
    floor_embedding_features = [*base_embedding_features, "floor_bucket"]
    base_features = list(e10.F18_FEATURES)
    return [
        {
            "experiment_name": "F18_reference_huber_010",
            "numeric_features": base_features,
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": base_embedding_features,
            "embedding_dims": base_embedding_dims,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 183,
            "loss": "huber_010",
        },
        {
            "experiment_name": "F31_floor_flags_huber_010",
            "numeric_features": [*base_features, *FLOOR_FLAG_FEATURES],
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": base_embedding_features,
            "embedding_dims": base_embedding_dims,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 311,
            "loss": "huber_010",
        },
        {
            "experiment_name": "F32_floor_bucket_huber_010",
            "numeric_features": [*base_features, *FLOOR_FLAG_FEATURES],
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": floor_embedding_features,
            "embedding_dims": floor_embedding_dims,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 322,
            "loss": "huber_010",
        },
        {
            "experiment_name": "F33_floor_interactions_huber_010",
            "numeric_features": [*base_features, *FLOOR_FLAG_FEATURES, *FLOOR_INTERACTION_FEATURES],
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": base_embedding_features,
            "embedding_dims": base_embedding_dims,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 333,
            "loss": "huber_010",
        },
        {
            "experiment_name": "F34_floor_full_huber_010",
            "numeric_features": [*base_features, *FLOOR_FLAG_FEATURES, *FLOOR_INTERACTION_FEATURES],
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": floor_embedding_features,
            "embedding_dims": floor_embedding_dims,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 344,
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
        "# F18 Floor Tuning Summary",
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
        "# F18 Floor Tuning Decision",
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
