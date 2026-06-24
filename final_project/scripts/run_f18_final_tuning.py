#!/usr/bin/env python3
from __future__ import annotations

import csv
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
from tensorflow import keras


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(os.environ.get("FINAL_PROJECT_DIR", "/Users/gwongwangjae/goorm-ai-language-course/final_project"))
E10_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_e10_outlier_signal_experiments.py"
RESIDUAL_FEATURE_PATH = PROJECT_DIR / "outputs" / "e11_region_residual_features.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "f18_final_tuning"
METRICS_CSV = OUTPUT_DIR / "f18_final_tuning_metrics.csv"
SUMMARY_MD = OUTPUT_DIR / "f18_final_tuning_summary.md"
FINAL_DECISION_MD = OUTPUT_DIR / "f18_final_tuning_final_decision.md"
SUCCESS_PATH = OUTPUT_DIR / "_SUCCESS"

RANDOM_STATE = 42
SPLIT_ORDER = ["train", "valid", "test", "recent_holdout"]
EVAL_SPLITS = ["valid", "test", "recent_holdout"]
LOSS_SPECS = [
    ("mse", "mse", 18),
    ("huber_005", "huber", 182),
    ("huber_0075", "huber", 185),
    ("huber_010", "huber", 183),
    ("huber_015", "huber", 186),
]
HUBER_DELTAS = {
    "huber_005": 0.05,
    "huber_0075": 0.075,
    "huber_010": 0.10,
    "huber_015": 0.15,
}
STATIC_BLEND_WEIGHTS = (0.25, 0.50, 0.75)
RISK_QUANTILES = (0.50, 0.70, 0.85)
CONFIDENCE_QUANTILES = (0.30, 0.50, 0.70)
RISK_COLUMNS = [
    "resid_expected_abs_pct_error",
    "best_resid_abs_pct_mean",
    "complex_resid_abs_pct_mean",
    "sgg_resid_abs_pct_mean",
    "complex_resid_error_gt_20_rate",
    "sgg_resid_error_gt_20_rate",
]
CONFIDENCE_COLUMNS = [
    "complex_resid_confidence",
    "sgg_resid_confidence",
]


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    epoch: int
    kind: str
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


def parse_epochs() -> list[int]:
    raw = os.environ.get("F18_TUNING_EPOCHS", "30,50,100")
    epochs = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            epochs.append(int(item))
    if not epochs:
        raise ValueError("F18_TUNING_EPOCHS is empty")
    return epochs


def load_e10_module():
    spec = importlib.util.spec_from_file_location("e10_outlier_runner", E10_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E10_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_modules(e10: Any, *, max_epochs: int) -> None:
    batch_size = int(os.environ.get("F18_TUNING_BATCH_SIZE", os.environ.get("E10_BATCH_SIZE", "8192")))
    patience = int(os.environ.get("F18_TUNING_EARLY_STOPPING_PATIENCE", "4"))
    verbose = int(os.environ.get("F18_TUNING_TRAIN_VERBOSE", "2"))
    e10.RUN_MODE = "full"
    e10.BATCH_SIZE = batch_size
    e10.MAX_EPOCHS = max_epochs
    e10.EARLY_STOPPING_PATIENCE = patience
    e10.TRAIN_VERBOSE = verbose
    e10.e09.RUN_MODE = "full"
    e10.e09.BATCH_SIZE = batch_size
    e10.e09.MAX_EPOCHS = max_epochs
    e10.e09.EARLY_STOPPING_PATIENCE = patience
    e10.e09.TRAIN_VERBOSE = verbose


def build_model(e10: Any, config: dict[str, Any], normalizer: Any, lookups: dict[str, Any]) -> keras.Model:
    tf.keras.utils.set_random_seed(RANDOM_STATE + int(config.get("seed_offset", 0)))
    numeric_input = keras.Input(shape=(len(e10.e09.numeric_features(config)),), name="numeric_input", dtype="float32")
    parts = [normalizer(numeric_input)]
    inputs = [numeric_input]
    for feature in e10.e09.embedding_features(config):
        inp = keras.Input(shape=(1,), name=f"{feature}_input", dtype=tf.string)
        idx = lookups[feature](inp)
        dim = int(config["embedding_dims"].get(feature, e10.e09.EMBEDDING_DIMS[feature]))
        emb = keras.layers.Embedding(lookups[feature].vocabulary_size(), dim, name=f"{feature}_embedding")(idx)
        inputs.append(inp)
        parts.append(keras.layers.Flatten(name=f"{feature}_flatten")(emb))
    x = keras.layers.Concatenate(name="feature_concat")(parts)
    for unit in config["dense_units"]:
        x = keras.layers.Dense(unit, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = keras.layers.Dropout(0.10 if unit >= 128 else 0.05)(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    loss_name = config["loss"]
    if loss_name == "mse":
        loss: Any = "mse"
    else:
        loss = keras.losses.Huber(delta=HUBER_DELTAS[loss_name])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


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

    if RESIDUAL_FEATURE_PATH.exists():
        header = pd.read_csv(RESIDUAL_FEATURE_PATH, nrows=0)
        residual_cols = ["transaction_id"] + [col for col in [*RISK_COLUMNS, *CONFIDENCE_COLUMNS] if col in header.columns]
        residual_df = pd.read_csv(RESIDUAL_FEATURE_PATH, usecols=residual_cols, dtype={"transaction_id": "string"})
        if not residual_df["transaction_id"].is_unique:
            raise RuntimeError("residual transaction_id is not unique")
        model_df = model_df.merge(residual_df, on="transaction_id", how="left", validate="one_to_one")

    model_df["trade_type"] = model_df["trade_type"].fillna("unknown")
    model_df = model_df.loc[(model_df["is_cancelled"] == 0) & (model_df["trade_type"].isin(["중개거래", "unknown"]))].copy()
    model_df = e10.add_e10_features(model_df)
    for col in RISK_COLUMNS:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").astype("float32")
            model_df[col] = model_df[col].fillna(float(model_df[col].median()))
    for col in CONFIDENCE_COLUMNS:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0).astype("float32")
    return model_df.sort_values(["deal_date", "transaction_id"]).reset_index(drop=True)


def config_for(e10: Any, *, name: str, loss: str, seed_offset: int) -> dict[str, Any]:
    return {
        "experiment_name": name,
        "numeric_features": list(e10.F18_FEATURES),
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e10.e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e10.e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": seed_offset,
        "loss": loss,
    }


def train_predictions(e10: Any, config: dict[str, Any], splits: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    tf.keras.backend.clear_session()
    print(f"\n=== {config['experiment_name']} ===", flush=True)
    medians = e10.e09.numeric_medians_for(config, splits)
    train_inputs, normalizer, lookups = e10.e09.build_preprocessors(config, splits["train"], medians)
    model = build_model(e10, config, normalizer, lookups)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=e10.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
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
    out: dict[str, np.ndarray] = {}
    for split_name in SPLIT_ORDER:
        raw_pred = model.predict(e10.e09.make_inputs(splits[split_name], config, medians), batch_size=e10.BATCH_SIZE, verbose=0).reshape(-1)
        out[split_name] = e10.e09.final_log_pred(splits[split_name], raw_pred, medians, config)
    return out


def metric_row(splits: dict[str, pd.DataFrame], split_name: str, pred_log: np.ndarray, candidate: str, epoch: int, kind: str) -> dict[str, Any]:
    split_df = splits[split_name]
    y_true = split_df["target"].to_numpy(dtype="float64")
    pred_log = np.asarray(pred_log, dtype="float64")
    pred_ppm = np.exp(pred_log)
    actual_ppm = split_df["price_per_m2"].to_numpy(dtype="float64")
    abs_pct = np.abs((pred_ppm - actual_ppm) / actual_ppm)
    abs_log = np.abs(pred_log - y_true)
    return {
        "candidate": candidate,
        "epoch": epoch,
        "kind": kind,
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
        "error_gt_30pct_rate": float((abs_pct > 0.30).mean()),
        "error_gt_50pct_rate": float((abs_pct > 0.50).mean()),
    }


def add_candidate_metrics(
    rows: list[dict[str, Any]],
    splits: dict[str, pd.DataFrame],
    candidate: str,
    epoch: int,
    kind: str,
    split_predictions: dict[str, np.ndarray],
) -> None:
    for split_name in EVAL_SPLITS:
        rows.append(metric_row(splits, split_name, split_predictions[split_name], candidate, epoch, kind))


def build_blend_candidates(
    predictions: dict[str, dict[str, np.ndarray]],
    splits: dict[str, pd.DataFrame],
    epoch: int,
) -> dict[str, tuple[str, dict[str, np.ndarray]]]:
    out: dict[str, tuple[str, dict[str, np.ndarray]]] = {}
    base_name = f"e{epoch}_mse"
    base = predictions[base_name]
    robust_names = [name for name in predictions if name.startswith(f"e{epoch}_huber")]
    for robust_name in robust_names:
        robust = predictions[robust_name]
        for weight in STATIC_BLEND_WEIGHTS:
            name = f"e{epoch}_{robust_name.split('_', 1)[1]}_blend_w{int(weight * 100):02d}"
            out[name] = (
                "static_blend",
                {split: weight * robust[split] + (1.0 - weight) * base[split] for split in EVAL_SPLITS},
            )
        for col in RISK_COLUMNS:
            if col not in splits["valid"].columns:
                continue
            valid_values = splits["valid"][col].to_numpy(dtype="float64")
            for quantile in RISK_QUANTILES:
                threshold = float(np.nanquantile(valid_values, quantile))
                name = f"e{epoch}_{robust_name.split('_', 1)[1]}_gate_{col}_q{int(quantile * 100)}"
                split_preds = {}
                for split in EVAL_SPLITS:
                    use_robust = splits[split][col].to_numpy(dtype="float64") <= threshold
                    split_preds[split] = np.where(use_robust, robust[split], base[split])
                out[name] = ("risk_gate", split_preds)
        for col in CONFIDENCE_COLUMNS:
            if col not in splits["valid"].columns:
                continue
            valid_values = splits["valid"][col].to_numpy(dtype="float64")
            for quantile in CONFIDENCE_QUANTILES:
                threshold = float(np.nanquantile(valid_values, quantile))
                name = f"e{epoch}_{robust_name.split('_', 1)[1]}_gate_{col}_q{int(quantile * 100)}"
                split_preds = {}
                for split in EVAL_SPLITS:
                    use_robust = splits[split][col].to_numpy(dtype="float64") >= threshold
                    split_preds[split] = np.where(use_robust, robust[split], base[split])
                out[name] = ("confidence_gate", split_preds)
    return out


def score_candidates(rows: list[dict[str, Any]], reference_name: str) -> list[CandidateScore]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], {})[row["split"]] = row
    ref = by_candidate[reference_name]
    scores: list[CandidateScore] = []
    for candidate, split_rows in by_candidate.items():
        if not all(split in split_rows for split in EVAL_SPLITS):
            continue
        valid = split_rows["valid"]
        test = split_rows["test"]
        recent = split_rows["recent_holdout"]
        valid_delta = float(valid["log_mae"] - ref["valid"]["log_mae"])
        test_delta = float(test["log_mae"] - ref["test"]["log_mae"])
        recent_delta = float(recent["log_mae"] - ref["recent_holdout"]["log_mae"])
        p99_delta = float(recent["abs_pct_error_p99"] - ref["recent_holdout"]["abs_pct_error_p99"])
        gt20_delta = float(recent["error_gt_20pct_rate"] - ref["recent_holdout"]["error_gt_20pct_rate"])
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
                candidate=candidate,
                epoch=int(recent["epoch"]),
                kind=str(recent["kind"]),
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


def write_outputs(rows: list[dict[str, Any]], scores: list[CandidateScore], reference_name: str) -> CandidateScore:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with METRICS_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
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
    top = sorted(scores, key=lambda score: (score.guardrail == "fail", score.recent_log_mae, score.recent_gt10, score.recent_p99))[:25]
    lines = [
        "# F18 Final Tuning Summary",
        "",
        "## Best",
        "",
        f"- reference: `{reference_name}`",
        f"- best: `{best.candidate}`",
        f"- kind: `{best.kind}`",
        f"- epoch: `{best.epoch}`",
        f"- guardrail: `{best.guardrail}`",
        "",
        "## Top Candidates",
        "",
        "| rank | candidate | kind | epoch | MAE | p95 | p99 | >10% | >20% | d_MAE | d_p99 | d_gt20 | guardrail |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, score in enumerate(top, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    score.candidate,
                    score.kind,
                    str(score.epoch),
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
        "# F18 Final Tuning Decision",
        "",
        "## Decision",
        "",
        f"- final candidate: `{best.candidate}`",
        f"- candidate kind: `{best.kind}`",
        f"- epoch: `{best.epoch}`",
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
    epochs = parse_epochs()
    e10 = load_e10_module()
    configure_modules(e10, max_epochs=max(epochs))
    print("project", PROJECT_DIR, flush=True)
    print("epochs", epochs, flush=True)
    print("output", OUTPUT_DIR, flush=True)
    model_df = load_model_frame(e10)
    splits = e10.e09.apply_smoke_sampling(e10.e09.split_frames(model_df))
    print(pd.DataFrame([{"split": name, "rows": len(frame)} for name, frame in splits.items()]), flush=True)

    rows: list[dict[str, Any]] = []
    all_scores: list[CandidateScore] = []
    reference_name = f"e{epochs[0]}_mse"
    for epoch in epochs:
        configure_modules(e10, max_epochs=epoch)
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for loss_name, loss_kind, seed_offset in LOSS_SPECS:
            candidate_name = f"e{epoch}_{loss_name}"
            config = config_for(e10, name=candidate_name, loss=loss_name, seed_offset=seed_offset)
            predictions[candidate_name] = train_predictions(e10, config, splits)
            add_candidate_metrics(rows, splits, candidate_name, epoch, loss_kind, predictions[candidate_name])
        for candidate_name, (kind, split_predictions) in build_blend_candidates(predictions, splits, epoch).items():
            add_candidate_metrics(rows, splits, candidate_name, epoch, kind, split_predictions)
        all_scores = score_candidates(rows, reference_name)
        write_outputs(rows, all_scores, reference_name)
        print(f"epoch_done {epoch}", flush=True)

    best = write_outputs(rows, all_scores, reference_name)
    print("best", best.candidate, best.kind, best.epoch, fmt(best.recent_log_mae), best.guardrail, flush=True)
    print("seconds", round(time.perf_counter() - start, 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
