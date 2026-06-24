#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
MONTHLY_RUNNER_PATH = REPO_ROOT / "scripts" / "run_f18_monthly_anchor_tuning.py"
NEWS_INPUT_PATHS = [
    REPO_ROOT / "apps/news/local-input/region-month-signal-bigkinds.csv.jsonl",
    REPO_ROOT / "apps/news/local-input/region-month-signal-web-research.jsonl",
]
OUTPUT_DIR = REPO_ROOT / "outputs" / "f18_news_tuning"
NEWS_SIDECAR = OUTPUT_DIR / "f18_news_region_month_features.csv"
METRICS_CSV = OUTPUT_DIR / "f18_news_tuning_metrics.csv"
SUMMARY_MD = OUTPUT_DIR / "f18_news_tuning_summary.md"
FINAL_DECISION_MD = OUTPUT_DIR / "f18_news_tuning_final_decision.md"
SUCCESS_PATH = OUTPUT_DIR / "_SUCCESS"

RANDOM_STATE = 42
EPOCH = int(os.environ.get("F18_NEWS_EPOCH", "30"))
REBUILD_NEWS = os.environ.get("F18_NEWS_REBUILD", "0") == "1"
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]
CANONICAL_RECENT = {
    "log_mae": 0.061775,
    "p95": 0.188077,
    "p99": 0.345582,
    "gt10": 0.177473,
    "gt20": 0.042952,
}

NEWS_SIGNAL_COLUMNS = [
    "news_count",
    "matched_news_count",
    "direct_evidence_count",
    "inherited_evidence_count",
    "policy_positive_score",
    "policy_negative_score",
    "redevelopment_score",
    "transport_score",
    "supply_risk_score",
    "sale_market_score",
    "rental_market_score",
    "price_up_signal",
    "price_down_signal",
    "confidence",
    "price_net_signal",
    "policy_net_score",
]
NEWS_FEATURES = [
    f"{scope}_{window}_{col}"
    for scope in ("national", "province", "detail")
    for window in ("lag1m", "lag3m")
    for col in NEWS_SIGNAL_COLUMNS
]

SGG_DETAIL_BUCKETS = {
    "11680": "SEOUL_GANGNAM_GU",
    "11650": "SEOUL_SEOCHO_GU",
    "11710": "SEOUL_SONGPA_GU",
    "11170": "SEOUL_YONGSAN_GU",
    "11440": "SEOUL_MAPO_GU",
    "11200": "SEOUL_SEONGDONG_GU",
    "11560": "SEOUL_YEONGDEUNGPO_GU",
    "11470": "SEOUL_YANGCHEON_GU",
    "11350": "SEOUL_NOWON_GU",
    "11740": "SEOUL_GANGDONG_GU",
}
GYEONGGI_PREFIX_BUCKETS = {
    "4111": "GYEONGGI_SUWON_SI",
    "4113": "GYEONGGI_SEONGNAM_SI",
    "4117": "GYEONGGI_ANYANG_SI",
    "4121": "GYEONGGI_GWANGMYEONG_SI",
    "4128": "GYEONGGI_GOYANG_SI",
    "4129": "GYEONGGI_GWACHEON_SI",
    "4136": "GYEONGGI_NAMYANGJU_SI",
    "4143": "GYEONGGI_UIWANG_SI",
    "4145": "GYEONGGI_HANAM_SI",
    "4146": "GYEONGGI_YONGIN_SI",
    "4157": "GYEONGGI_GIMPO_SI",
    "4159": "GYEONGGI_HWASEONG_SI",
}


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(monthly: Any, e10: Any) -> None:
    monthly.EPOCH = EPOCH
    batch_size = int(os.environ.get("F18_NEWS_BATCH_SIZE", os.environ.get("E10_BATCH_SIZE", "8192")))
    patience = int(os.environ.get("F18_NEWS_EARLY_STOPPING_PATIENCE", "4"))
    verbose = int(os.environ.get("F18_NEWS_TRAIN_VERBOSE", "2"))
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


def month_idx(value: str) -> int:
    year, month = value.split("-")
    return int(year) * 12 + int(month)


def read_news_signals() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in NEWS_INPUT_PATHS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row = {
                    "region_bucket": raw["region_bucket"],
                    "signal_month_idx": month_idx(raw["signal_month"]),
                }
                for col in NEWS_SIGNAL_COLUMNS:
                    if col == "price_net_signal":
                        value = float(raw.get("price_up_signal", 0)) - float(raw.get("price_down_signal", 0))
                    elif col == "policy_net_score":
                        value = float(raw.get("policy_positive_score", 0)) - float(raw.get("policy_negative_score", 0))
                    else:
                        value = float(raw.get(col, 0))
                    row[col] = value
                rows.append(row)
    if not rows:
        raise RuntimeError("no news signal rows found")
    df = pd.DataFrame(rows)
    df = (
        df.sort_values(["signal_month_idx", "region_bucket"])
        .drop_duplicates(["signal_month_idx", "region_bucket"], keep="last")
        .reset_index(drop=True)
    )
    return df


def detail_bucket_for_sgg(sgg_code: object) -> str:
    code = str(sgg_code)[:5]
    if code in SGG_DETAIL_BUCKETS:
        return SGG_DETAIL_BUCKETS[code]
    prefix = code[:4]
    return GYEONGGI_PREFIX_BUCKETS.get(prefix, "OTHER")


def province_bucket_for_sgg(sgg_code: object) -> str:
    code = str(sgg_code)
    if code.startswith("11"):
        return "SEOUL"
    if code.startswith("41"):
        return "GYEONGGI"
    return "OTHER"


def lag_table(news: pd.DataFrame, scope: str) -> pd.DataFrame:
    lag1 = news[["region_bucket", "signal_month_idx", *NEWS_SIGNAL_COLUMNS]].copy()
    lag1["target_month_idx"] = lag1["signal_month_idx"] + 1
    lag1 = lag1.drop(columns=["signal_month_idx"]).rename(columns={col: f"{scope}_lag1m_{col}" for col in NEWS_SIGNAL_COLUMNS})

    parts = []
    for shift in (1, 2, 3):
        part = news[["region_bucket", "signal_month_idx", *NEWS_SIGNAL_COLUMNS]].copy()
        part["target_month_idx"] = part["signal_month_idx"] + shift
        parts.append(part.drop(columns=["signal_month_idx"]))
    lag3 = pd.concat(parts, ignore_index=True)
    lag3 = (
        lag3.groupby(["region_bucket", "target_month_idx"], dropna=False, observed=True)[NEWS_SIGNAL_COLUMNS]
        .mean()
        .reset_index()
        .rename(columns={col: f"{scope}_lag3m_{col}" for col in NEWS_SIGNAL_COLUMNS})
    )
    return lag1.merge(lag3, on=["region_bucket", "target_month_idx"], how="outer", validate="one_to_one")


def add_scope_features(base: pd.DataFrame, news: pd.DataFrame, scope: str, bucket_col: str) -> pd.DataFrame:
    scoped = lag_table(news, scope).rename(columns={"region_bucket": bucket_col})
    return base.merge(scoped, on=[bucket_col, "target_month_idx"], how="left", validate="many_to_one")


def build_news_sidecar(e10: Any) -> None:
    if NEWS_SIDECAR.exists() and not REBUILD_NEWS:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        e10.DATA_PATH,
        usecols=["transaction_id", "sgg_code", "deal_date"],
        dtype={"transaction_id": "string", "sgg_code": "string"},
        parse_dates=["deal_date"],
    )
    if not raw["transaction_id"].is_unique:
        raise RuntimeError("transactions.csv transaction_id is not unique")
    raw["target_month_idx"] = (raw["deal_date"].dt.year * 12 + raw["deal_date"].dt.month).astype("int32")
    raw["national_bucket"] = "NATIONAL"
    raw["province_bucket"] = raw["sgg_code"].map(province_bucket_for_sgg).astype("string")
    raw["detail_bucket"] = raw["sgg_code"].map(detail_bucket_for_sgg).astype("string")
    base = raw[["transaction_id", "target_month_idx", "national_bucket", "province_bucket", "detail_bucket"]].copy()

    news = read_news_signals()
    features = add_scope_features(base, news, "national", "national_bucket")
    features = add_scope_features(features, news, "province", "province_bucket")
    features = add_scope_features(features, news, "detail", "detail_bucket")
    keep_cols = ["transaction_id", *NEWS_FEATURES]
    for col in NEWS_FEATURES:
        if col not in features.columns:
            features[col] = np.nan
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0).astype("float32")
        if col.endswith(("news_count", "matched_news_count", "direct_evidence_count", "inherited_evidence_count")):
            features[col] = np.log1p(features[col].clip(lower=0)).astype("float32")
        elif col.endswith("confidence"):
            features[col] = features[col].clip(lower=0, upper=1).astype("float32")
        else:
            features[col] = (features[col] / 100.0).astype("float32")
    features[keep_cols].to_csv(NEWS_SIDECAR, index=False)


def load_model_frame(monthly: Any, e10: Any) -> pd.DataFrame:
    build_news_sidecar(e10)
    model_df = monthly.load_model_frame(e10)
    sidecar = pd.read_csv(NEWS_SIDECAR, dtype={"transaction_id": "string"})
    if not sidecar["transaction_id"].is_unique:
        raise RuntimeError("news sidecar transaction_id is not unique")
    out = model_df.merge(sidecar, on="transaction_id", how="left", validate="one_to_one")
    for col in NEWS_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("float32")
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
            "experiment_name": "F39_news_region_month_huber_010",
            "numeric_features": [*base_features, *NEWS_FEATURES],
            "base_log_feature": "log_complex_prev_price_per_m2",
            "embedding_features": e10.e09.BASE_EMBEDDING_FEATURES,
            "embedding_dims": e10.e09.EMBEDDING_DIMS,
            "learning_rate": 0.001,
            "dense_units": [128, 64],
            "seed_offset": 391,
            "loss": "huber_010",
        },
    ]


def score_row(metrics: pd.DataFrame, candidate: str, split: str) -> pd.Series:
    rows = metrics.loc[(metrics["experiment_name"] == candidate) & (metrics["split"] == split)]
    if rows.empty:
        raise RuntimeError(f"missing metric row: {candidate} {split}")
    return rows.iloc[0]


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def fmt(value: float) -> str:
    return f"{value:.6f}"


def write_outputs(metrics: pd.DataFrame) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_CSV, index=False)
    ref = score_row(metrics, "F18_reference_huber_010", "recent_holdout")
    news = score_row(metrics, "F39_news_region_month_huber_010", "recent_holdout")
    d_mae = float(news["log_mae"] - ref["log_mae"])
    d_p99 = float(news["abs_pct_error_p99"] - ref["abs_pct_error_p99"])
    d_gt20 = float(news["error_gt_20pct_rate"] - ref["error_gt_20pct_rate"])
    beats_same_run = d_mae < -0.0001 and d_p99 <= 0.003 and d_gt20 <= 0.001
    beats_canonical = (
        float(news["log_mae"]) < CANONICAL_RECENT["log_mae"]
        and float(news["abs_pct_error_p95"]) <= CANONICAL_RECENT["p95"] + 0.001
        and float(news["abs_pct_error_p99"]) <= CANONICAL_RECENT["p99"] + 0.003
        and float(news["error_gt_10pct_rate"]) <= CANONICAL_RECENT["gt10"] + 0.001
        and float(news["error_gt_20pct_rate"]) <= CANONICAL_RECENT["gt20"] + 0.001
    )
    final = "F39_news_region_month_huber_010" if beats_same_run and beats_canonical else "canonical_F18_reference_huber_010"

    rows = []
    for candidate in ("F18_reference_huber_010", "F39_news_region_month_huber_010"):
        recent = score_row(metrics, candidate, "recent_holdout")
        rows.append(
            "| "
            + " | ".join(
                [
                    candidate,
                    fmt(float(recent["log_mae"])),
                    pct(float(recent["abs_pct_error_p95"])),
                    pct(float(recent["abs_pct_error_p99"])),
                    pct(float(recent["error_gt_10pct_rate"])),
                    pct(float(recent["error_gt_20pct_rate"])),
                ]
            )
            + " |"
        )
    rows.append("| canonical_F18_reference_huber_010 | 0.061775 | 18.8077% | 34.5582% | 17.7473% | 4.2952% |")

    summary = [
        "# F18 News Tuning Summary",
        "",
        f"- epoch: `{EPOCH}`",
        f"- news feature: `lag1m`, `lag3m`, scopes `national/province/detail`",
        f"- final: `{final}`",
        "",
        "| candidate | recent MAE | p95 | p99 | >10% | >20% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Decision Checks",
        "",
        f"- same-run news d_MAE: `{fmt(d_mae)}`",
        f"- same-run news d_p99: `{pct(d_p99)}`",
        f"- same-run news d_>20%: `{pct(d_gt20)}`",
        f"- beats_same_run: `{str(beats_same_run).lower()}`",
        f"- beats_canonical: `{str(beats_canonical).lower()}`",
    ]
    SUMMARY_MD.write_text("\n".join(summary) + "\n", encoding="utf-8")
    FINAL_DECISION_MD.write_text(
        "\n".join(
            [
                "# F18 News Tuning Decision",
                "",
                f"- final candidate: `{final}`",
                f"- news candidate: `F39_news_region_month_huber_010`",
                f"- beats same-run F18: `{str(beats_same_run).lower()}`",
                f"- beats canonical F18: `{str(beats_canonical).lower()}`",
                f"- news recent_holdout MAE: `{fmt(float(news['log_mae']))}`",
                f"- news recent_holdout p95: `{pct(float(news['abs_pct_error_p95']))}`",
                f"- news recent_holdout p99: `{pct(float(news['abs_pct_error_p99']))}`",
                f"- news recent_holdout >10%: `{pct(float(news['error_gt_10pct_rate']))}`",
                f"- news recent_holdout >20%: `{pct(float(news['error_gt_20pct_rate']))}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    SUCCESS_PATH.write_text(f"completed_at_utc={pd.Timestamp.utcnow().isoformat()}\n", encoding="utf-8")
    return final


def main() -> int:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    monthly = import_module(MONTHLY_RUNNER_PATH, "f18_monthly_runner_for_news")
    e10 = monthly.load_e10_module()
    configure(monthly, e10)
    print("epoch", EPOCH, flush=True)
    print("output", OUTPUT_DIR, flush=True)
    model_df = load_model_frame(monthly, e10)
    splits = e10.e09.apply_smoke_sampling(e10.e09.split_frames(model_df))
    print(pd.DataFrame([{"split": name, "rows": len(frame)} for name, frame in splits.items()]), flush=True)
    metric_frames = []
    for config in experiment_configs(e10):
        print("\n===", config["experiment_name"], "===", flush=True)
        metrics, _groups = e10.e09.train_and_predict(config, splits)
        metric_frames.append(metrics)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    final = write_outputs(metrics_df)
    print("final", final, flush=True)
    print("seconds", round(time.perf_counter() - start, 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
