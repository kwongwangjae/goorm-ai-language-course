#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

import sweep_f18_canonical_deployment_runs as f18_sweep


PROJECT_DIR = Path(__file__).resolve().parents[1]
F29_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_e11_region_residual_experiments.py"
F36_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_f18_monthly_anchor_tuning.py"
F37_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_f18_prev3_tuning.py"
PROGRESSIVE_RUNNER_PATH = PROJECT_DIR / "scripts" / "run_f18_progressive_tuning.py"
RUNS_DIR = PROJECT_DIR / "models" / "f29_f36_deployment_runs"
BEST_DIR = PROJECT_DIR / "models" / "best_price_deployment_attempt"
SUMMARY_CSV = RUNS_DIR / "sweep_summary.csv"
SUMMARY_MD = RUNS_DIR / "sweep_summary.md"

HISTORICAL_RECENT = f18_sweep.HISTORICAL_RECENT


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def run_f29() -> pd.DataFrame:
    f29 = load_module(F29_RUNNER_PATH, "f29_deployment_runner")
    f18_sweep.RUNS_DIR = RUNS_DIR
    config = next(item for item in f29.EXPERIMENTS if item["experiment_name"] == "F29_residual_bias_features_huber")
    config = dict(config)
    config["run_name"] = "F29_residual_bias_features_huber"
    config["model_version"] = "deployment__F29_residual_bias_features_huber"
    frame = f29.load_model_frame()
    splits = f18_sweep.artifact.split_frames(frame)
    return f18_sweep.train_one(config, splits)


def run_f36() -> pd.DataFrame:
    f36 = load_module(F36_RUNNER_PATH, "f36_deployment_runner")
    f18_sweep.RUNS_DIR = RUNS_DIR
    e10 = f36.load_e10_module()
    f36.configure_modules(e10)
    f36.build_monthly_sidecar(e10)
    config = next(item for item in f36.experiment_configs(e10) if item["experiment_name"] == "F36_monthly_market_anchor_huber_010")
    config = dict(config)
    config["run_name"] = "F36_monthly_market_anchor_huber_010"
    config["model_version"] = "deployment__F36_monthly_market_anchor_huber_010"
    frame = f36.load_model_frame(e10)
    splits = f18_sweep.artifact.split_frames(frame)
    return f18_sweep.train_one(config, splits)


def f18_base_config(name: str, features: list[str], e10, seed_offset: int) -> dict:
    return {
        "experiment_name": name,
        "numeric_features": features,
        "base_log_feature": "log_complex_prev_price_per_m2",
        "embedding_features": e10.e09.BASE_EMBEDDING_FEATURES,
        "embedding_dims": e10.e09.EMBEDDING_DIMS,
        "learning_rate": 0.001,
        "dense_units": [128, 64],
        "seed_offset": seed_offset,
        "loss": "huber_010",
        "run_name": name,
        "model_version": f"deployment__{name}",
    }


def run_f37() -> pd.DataFrame:
    f36 = load_module(F36_RUNNER_PATH, "f36_deployment_runner_f37")
    prev3 = load_module(F37_RUNNER_PATH, "f37_prev3_runner")
    progressive = load_module(PROGRESSIVE_RUNNER_PATH, "f37_progressive_runner")
    f18_sweep.RUNS_DIR = RUNS_DIR
    e10 = f36.load_e10_module()
    f36.configure_modules(e10)
    frame = f36.load_model_frame(e10)
    frame = progressive.add_prev3_to_frame(prev3, e10, frame)
    features = [*list(e10.F18_FEATURES), *f36.MONTHLY_FEATURES, *prev3.PREV3_FEATURES]
    config = f18_base_config("F37_monthly_anchor_prev3_rolling_huber_010", features, e10, 371)
    splits = f18_sweep.artifact.split_frames(frame)
    return f18_sweep.train_one(config, splits)


def run_f38() -> pd.DataFrame:
    f36 = load_module(F36_RUNNER_PATH, "f36_deployment_runner_f38")
    progressive = load_module(PROGRESSIVE_RUNNER_PATH, "f38_progressive_runner")
    f18_sweep.RUNS_DIR = RUNS_DIR
    e10 = f36.load_e10_module()
    f36.configure_modules(e10)
    frame = f36.load_model_frame(e10)
    frame = progressive.add_sparse_gap_features(frame)
    features = [*list(e10.F18_FEATURES), *f36.MONTHLY_FEATURES, *progressive.SPARSE_GAP_FEATURES]
    config = f18_base_config("F38_monthly_sparse_gap_huber_010", features, e10, 381)
    splits = f18_sweep.artifact.split_frames(frame)
    return f18_sweep.train_one(config, splits)


def write_summary(all_metrics: pd.DataFrame) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(RUNS_DIR / "all_eval_metrics.csv", index=False)
    recent = all_metrics.loc[all_metrics["split"].eq("recent_holdout")].copy()
    recent["delta_mae_vs_historical"] = recent["log_mae"] - HISTORICAL_RECENT["log_mae"]
    recent = recent.sort_values(["log_mae", "error_gt_20pct_rate", "abs_pct_error_p99"]).reset_index(drop=True)
    recent.to_csv(SUMMARY_CSV, index=False)
    best = recent.iloc[0]
    best_run = str(best["model_version"]).replace("deployment__", "")
    source_dir = RUNS_DIR / best_run
    if BEST_DIR.exists():
        shutil.rmtree(BEST_DIR)
    shutil.copytree(source_dir, BEST_DIR)
    lines = [
        "# F29/F36 deployment candidate sweep",
        "",
        "## Best saved attempt",
        f"- run: `{best_run}`",
        f"- artifact: `{BEST_DIR}`",
        f"- recent_holdout MAE(log): `{best['log_mae']:.6f}`",
        f"- delta MAE vs historical: `{best['delta_mae_vs_historical']:.6f}`",
        "",
        "## Ranking",
        "| rank | run | MAE(log) | p95 | p99 | >10% | >20% | d_MAE_hist |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in recent.iterrows():
        run = str(row["model_version"]).replace("deployment__", "")
        lines.append(
            f"| {idx + 1} | {run} | {row['log_mae']:.6f} | {pct(row['abs_pct_error_p95'])} | {pct(row['abs_pct_error_p99'])} | "
            f"{pct(row['error_gt_10pct_rate'])} | {pct(row['error_gt_20pct_rate'])} | {row['delta_mae_vs_historical']:.6f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    frames = [run_f29(), run_f36(), run_f37(), run_f38()]
    write_summary(pd.concat(frames, ignore_index=True))
    print("summary", SUMMARY_MD)
    print("best", BEST_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
