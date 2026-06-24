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
OUTPUT_DIR = REPO_ROOT / "outputs" / "f18_progressive_tuning"
METRICS_CSV = OUTPUT_DIR / "f18_progressive_tuning_metrics.csv"
AGG_CSV = OUTPUT_DIR / "f18_progressive_tuning_aggregate.csv"
SUMMARY_MD = OUTPUT_DIR / "f18_progressive_tuning_summary.md"
FINAL_DECISION_MD = OUTPUT_DIR / "f18_progressive_tuning_final_decision.md"
SUCCESS_PATH = OUTPUT_DIR / "_SUCCESS"

MONTHLY_RUNNER_PATH = REPO_ROOT / "scripts" / "run_f18_monthly_anchor_tuning.py"
PREV3_RUNNER_PATH = REPO_ROOT / "scripts" / "run_f18_prev3_tuning.py"

RANDOM_STATE = 42
EPOCH = int(os.environ.get("F18_PROGRESSIVE_EPOCH", "30"))
SEEDS = [int(item.strip()) for item in os.environ.get("F18_PROGRESSIVE_SEEDS", "183,184,185").split(",") if item.strip()]
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]

BASELINE = "F18_reference_huber_010"
MONTHLY = "F36_monthly_market_anchor_huber_010"
MONTHLY_PREV3 = "F37_monthly_anchor_prev3_rolling_huber_010"
MONTHLY_SPARSE = "F38_monthly_sparse_gap_huber_010"

SPARSE_GAP_FEATURES = [
    "prev_gap_gt_6m",
    "prev_gap_gt_12m",
    "prev_gap_gt_24m",
    "exact_prev_missing_count",
    "monthly_anchor_missing_count",
    "prev_gap_x_complex_lag3_missing",
    "prev_gap_x_exact_lag3_missing",
    "prev1_vs_complex_lag3_log_gap_abs",
    "exact_prev1_vs_exact_lag3_log_gap_abs",
    "complex_lag3m_count_log1p",
    "exact_area_lag3m_count_log1p",
    "sgg_lag3m_count_log1p",
]


@dataclass(frozen=True)
class StageDecision:
    stage: str
    candidate: str
    reference: str
    action: str
    reason: str


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(e10: Any) -> None:
    batch_size = int(os.environ.get("F18_PROGRESSIVE_BATCH_SIZE", os.environ.get("E10_BATCH_SIZE", "8192")))
    patience = int(os.environ.get("F18_PROGRESSIVE_EARLY_STOPPING_PATIENCE", "4"))
    verbose = int(os.environ.get("F18_PROGRESSIVE_TRAIN_VERBOSE", "2"))
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


def metric_row(split_df: pd.DataFrame, pred_log: np.ndarray, candidate: str, seed: int, split_name: str) -> dict[str, Any]:
    y_true = split_df["target"].to_numpy(dtype="float64")
    pred_log = np.asarray(pred_log, dtype="float64").reshape(-1)
    pred_ppm = np.exp(pred_log)
    actual_ppm = split_df["price_per_m2"].to_numpy(dtype="float64")
    abs_pct = np.abs((pred_ppm - actual_ppm) / actual_ppm)
    abs_log = np.abs(pred_log - y_true)
    return {
        "candidate": candidate,
        "seed": seed,
        "epoch": EPOCH,
        "split": split_name,
        "rows": len(split_df),
        "log_mae": float(abs_log.mean()),
        "abs_pct_error_p95": float(np.quantile(abs_pct, 0.95)),
        "abs_pct_error_p99": float(np.quantile(abs_pct, 0.99)),
        "error_gt_10pct_rate": float((abs_pct > 0.10).mean()),
        "error_gt_20pct_rate": float((abs_pct > 0.20).mean()),
    }


def train_candidate(e10: Any, config: dict[str, Any], splits: dict[str, pd.DataFrame], seed: int) -> list[dict[str, Any]]:
    tf.keras.backend.clear_session()
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE + seed)
    config = {**config, "seed_offset": seed}
    print(f"\n=== {config['experiment_name']} seed={seed} ===", flush=True)
    medians = e10.e09.numeric_medians_for(config, splits)
    train_inputs, normalizer, lookups = e10.e09.build_preprocessors(config, splits["train"], medians)
    model = e10.build_model(config, normalizer, lookups)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=e10.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    start = time.perf_counter()
    model.fit(
        train_inputs,
        e10.e09.y_for(splits["train"], medians, config),
        validation_data=(e10.e09.make_inputs(splits["valid"], config, medians), e10.e09.y_for(splits["valid"], medians, config)),
        epochs=e10.MAX_EPOCHS,
        batch_size=e10.BATCH_SIZE,
        callbacks=callbacks,
        verbose=e10.TRAIN_VERBOSE,
    )
    print("duration_seconds", round(time.perf_counter() - start, 2), flush=True)
    rows = []
    for split_name in EVAL_SPLITS:
        raw_pred = model.predict(e10.e09.make_inputs(splits[split_name], config, medians), batch_size=e10.BATCH_SIZE, verbose=0).reshape(-1)
        pred_log = e10.e09.final_log_pred(splits[split_name], raw_pred, medians, config)
        rows.append(metric_row(splits[split_name], pred_log, config["experiment_name"], seed, split_name))
    return rows


def add_prev3_to_frame(prev3: Any, e10: Any, frame: pd.DataFrame) -> pd.DataFrame:
    prev3.build_prev3_sidecar(e10)
    sidecar = pd.read_csv(
        prev3.PREV3_SIDECAR,
        dtype={"transaction_id": "string", "prev3_missing": "Int8", "exact_prev3_missing": "Int8"},
        parse_dates=["prev3_source_deal_date", "exact_prev3_source_deal_date"],
    )
    out = frame.merge(sidecar, on="transaction_id", how="left", validate="one_to_one")
    return prev3.add_prev3_features(out)


def add_sparse_gap_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    prev_gap = pd.to_numeric(out["prev_deal_gap_months"], errors="coerce").fillna(999).astype("float32")
    out["prev_gap_gt_6m"] = (prev_gap > 6).astype("float32")
    out["prev_gap_gt_12m"] = (prev_gap > 12).astype("float32")
    out["prev_gap_gt_24m"] = (prev_gap > 24).astype("float32")
    out["exact_prev_missing_count"] = (
        pd.to_numeric(out["exact_prev1_missing"], errors="coerce").fillna(1)
        + pd.to_numeric(out["exact_prev2_missing"], errors="coerce").fillna(1)
    ).astype("float32")
    out["monthly_anchor_missing_count"] = (
        pd.to_numeric(out["complex_lag3m_missing"], errors="coerce").fillna(1)
        + pd.to_numeric(out["exact_area_lag3m_missing"], errors="coerce").fillna(1)
        + pd.to_numeric(out["sgg_lag3m_missing"], errors="coerce").fillna(1)
    ).astype("float32")
    out["prev_gap_x_complex_lag3_missing"] = (prev_gap * pd.to_numeric(out["complex_lag3m_missing"], errors="coerce").fillna(1)).astype("float32")
    out["prev_gap_x_exact_lag3_missing"] = (prev_gap * pd.to_numeric(out["exact_area_lag3m_missing"], errors="coerce").fillna(1)).astype("float32")
    out["prev1_vs_complex_lag3_log_gap_abs"] = pd.to_numeric(out["prev1_vs_complex_lag3m_log_gap"], errors="coerce").abs().astype("float32")
    out["exact_prev1_vs_exact_lag3_log_gap_abs"] = pd.to_numeric(out["exact_prev1_vs_exact_lag3m_log_gap"], errors="coerce").abs().astype("float32")
    for col in ["complex_lag3m_count", "exact_area_lag3m_count", "sgg_lag3m_count"]:
        out[f"{col}_log1p"] = np.log1p(pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)).astype("float32")
    return out


def config(name: str, features: list[str], e10: Any, seed: int) -> dict[str, Any]:
    return {
        "experiment_name": name,
        "numeric_features": features,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e10.e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e10.e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": seed,
        "loss": "huber_010",
    }


def aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    recent = df.loc[df["split"] == "recent_holdout"]
    agg = (
        recent.groupby("candidate", sort=False)
        .agg(
            runs=("seed", "nunique"),
            mean_log_mae=("log_mae", "mean"),
            std_log_mae=("log_mae", "std"),
            mean_p95=("abs_pct_error_p95", "mean"),
            mean_p99=("abs_pct_error_p99", "mean"),
            mean_gt10=("error_gt_10pct_rate", "mean"),
            mean_gt20=("error_gt_20pct_rate", "mean"),
        )
        .reset_index()
    )
    return agg


def improved(candidate: pd.Series, reference: pd.Series) -> tuple[bool, str]:
    d_mae = float(candidate["mean_log_mae"] - reference["mean_log_mae"])
    d_p99 = float(candidate["mean_p99"] - reference["mean_p99"])
    d_gt20 = float(candidate["mean_gt20"] - reference["mean_gt20"])
    if d_mae <= -0.0001:
        return True, f"mean MAE improved by {-d_mae:.6f}"
    if d_mae <= 0.0001 and (d_p99 <= -0.003 or d_gt20 <= -0.001):
        return True, f"MAE tied and tail improved p99={d_p99:.6f}, gt20={d_gt20:.6f}"
    return False, f"no sufficient improvement: d_MAE={d_mae:.6f}, d_p99={d_p99:.6f}, d_gt20={d_gt20:.6f}"


def agg_row(agg: pd.DataFrame, name: str) -> pd.Series:
    matches = agg.loc[agg["candidate"] == name]
    if matches.empty:
        raise RuntimeError(f"missing aggregate row: {name}")
    return matches.iloc[0]


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def write_outputs(rows: list[dict[str, Any]], decisions: list[StageDecision]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    agg = aggregate(rows)
    metrics.to_csv(METRICS_CSV, index=False)
    agg.to_csv(AGG_CSV, index=False)
    ordered = agg.sort_values(["mean_log_mae", "mean_gt10", "mean_p99"]).reset_index(drop=True)
    lines = [
        "# F18 Progressive Tuning Summary",
        "",
        f"- epoch: `{EPOCH}`",
        f"- seeds: `{','.join(map(str, SEEDS))}`",
        "",
        "## Recent Holdout Mean Ranking",
        "",
        "| rank | candidate | runs | MAE_mean | MAE_std | p95_mean | p99_mean | >10%_mean | >20%_mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in ordered.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx + 1),
                    str(row["candidate"]),
                    str(int(row["runs"])),
                    f"{row['mean_log_mae']:.6f}",
                    "" if pd.isna(row["std_log_mae"]) else f"{row['std_log_mae']:.6f}",
                    pct(float(row["mean_p95"])),
                    pct(float(row["mean_p99"])),
                    pct(float(row["mean_gt10"])),
                    pct(float(row["mean_gt20"])),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Stage Decisions",
        "",
        "| stage | candidate | reference | action | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for decision in decisions:
        lines.append(f"| {decision.stage} | {decision.candidate} | {decision.reference} | {decision.action} | {decision.reason} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = ordered.iloc[0]
    final_lines = [
        "# F18 Progressive Tuning Decision",
        "",
        f"- best candidate: `{best['candidate']}`",
        f"- epoch: `{EPOCH}`",
        f"- seeds: `{','.join(map(str, SEEDS))}`",
        f"- recent_holdout mean log_mae: `{best['mean_log_mae']:.6f}`",
        f"- recent_holdout mean p95: `{pct(float(best['mean_p95']))}`",
        f"- recent_holdout mean p99: `{pct(float(best['mean_p99']))}`",
        f"- recent_holdout mean >10%: `{pct(float(best['mean_gt10']))}`",
        f"- recent_holdout mean >20%: `{pct(float(best['mean_gt20']))}`",
        "",
        "## Decisions",
    ]
    for decision in decisions:
        final_lines.append(f"- {decision.stage}: `{decision.action}` - {decision.reason}")
    FINAL_DECISION_MD.write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    SUCCESS_PATH.write_text(f"completed_at_utc={pd.Timestamp.utcnow().isoformat()}\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly = import_module(MONTHLY_RUNNER_PATH, "f18_monthly_anchor_runner")
    prev3 = import_module(PREV3_RUNNER_PATH, "f18_prev3_runner")
    e10 = monthly.load_e10_module()
    configure(e10)
    print("epoch", EPOCH, flush=True)
    print("seeds", SEEDS, flush=True)
    print("output", OUTPUT_DIR, flush=True)

    model_df = monthly.load_model_frame(e10)
    combo_df = add_prev3_to_frame(prev3, e10, model_df)
    sparse_df = add_sparse_gap_features(model_df)

    base_features = list(e10.F18_FEATURES)
    monthly_features = [*base_features, *monthly.MONTHLY_FEATURES]
    combo_features = [*base_features, *monthly.MONTHLY_FEATURES, *prev3.PREV3_FEATURES]
    sparse_features = [*base_features, *monthly.MONTHLY_FEATURES, *SPARSE_GAP_FEATURES]

    rows: list[dict[str, Any]] = []
    decisions: list[StageDecision] = []

    splits = e10.e09.apply_smoke_sampling(e10.e09.split_frames(model_df))
    print(pd.DataFrame([{"split": name, "rows": len(frame)} for name, frame in splits.items()]), flush=True)
    for seed in SEEDS:
        rows.extend(train_candidate(e10, config(BASELINE, base_features, e10, seed), splits, seed))
    decisions.append(StageDecision("1_baseline_repro", BASELINE, "-", "complete", "baseline repeated across seeds"))
    write_outputs(rows, decisions)

    for seed in SEEDS:
        rows.extend(train_candidate(e10, config(MONTHLY, monthly_features, e10, seed), splits, seed))
    agg = aggregate(rows)
    ok, reason = improved(agg_row(agg, MONTHLY), agg_row(agg, BASELINE))
    decisions.append(StageDecision("2_monthly_anchor", MONTHLY, BASELINE, "continue" if ok else "stop", reason))
    write_outputs(rows, decisions)
    if not ok:
        print("stop_after_stage", 2, reason, flush=True)
        print("seconds", round(time.perf_counter() - start, 2), flush=True)
        return 0

    combo_splits = e10.e09.apply_smoke_sampling(e10.e09.split_frames(combo_df))
    for seed in SEEDS:
        rows.extend(train_candidate(e10, config(MONTHLY_PREV3, combo_features, e10, seed), combo_splits, seed))
    agg = aggregate(rows)
    ok_combo, reason_combo = improved(agg_row(agg, MONTHLY_PREV3), agg_row(agg, MONTHLY))
    decisions.append(StageDecision("3_monthly_prev3", MONTHLY_PREV3, MONTHLY, "continue" if ok_combo else "no_adoption_continue_to_independent_stage4", reason_combo))
    write_outputs(rows, decisions)

    sparse_splits = e10.e09.apply_smoke_sampling(e10.e09.split_frames(sparse_df))
    for seed in SEEDS:
        rows.extend(train_candidate(e10, config(MONTHLY_SPARSE, sparse_features, e10, seed), sparse_splits, seed))
    agg = aggregate(rows)
    ok_sparse, reason_sparse = improved(agg_row(agg, MONTHLY_SPARSE), agg_row(agg, MONTHLY))
    decisions.append(StageDecision("4_monthly_sparse_gap", MONTHLY_SPARSE, MONTHLY, "adopt_candidate" if ok_sparse else "stop_no_adoption", reason_sparse))
    write_outputs(rows, decisions)

    print("best", agg.sort_values(["mean_log_mae", "mean_gt10", "mean_p99"]).iloc[0]["candidate"], flush=True)
    print("seconds", round(time.perf_counter() - start, 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
