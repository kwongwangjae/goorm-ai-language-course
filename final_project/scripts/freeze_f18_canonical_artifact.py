#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
FINAL_PROJECT_DIR = Path("/Users/gwongwangjae/goorm-ai-language-course/final_project")
SOURCE_OUTPUT_DIR = FINAL_PROJECT_DIR / "outputs"
SOURCE_SCRIPT_DIR = FINAL_PROJECT_DIR / "scripts"
ARTIFACT_DIR = REPO_DIR / "outputs" / "f18_canonical_model_artifact"
EVIDENCE_DIR = ARTIFACT_DIR / "evidence"

FINAL_MODEL = "canonical_F18_reference_huber_010"
SOURCE_EXPERIMENT = "F18_reference_huber_010"
REFERENCE_EXPERIMENT = "F18_reference_recheck"

EVIDENCE_FILES = [
    SOURCE_OUTPUT_DIR / "e10_outlier_signal_metrics.csv",
    SOURCE_OUTPUT_DIR / "e10_outlier_signal_group_metrics.csv",
    SOURCE_OUTPUT_DIR / "e10_outlier_signal_summary.md",
    SOURCE_OUTPUT_DIR / "e10_outlier_signal_final_decision.md",
    SOURCE_OUTPUT_DIR / "e10_outlier_signal_feature_quality_report.md",
]

HASH_ONLY_FILES = [
    SOURCE_OUTPUT_DIR / "e10_outlier_signal_features.csv",
    SOURCE_SCRIPT_DIR / "run_e10_outlier_signal_experiments.py",
    SOURCE_SCRIPT_DIR / "run_e09_exact_prev_experiments.py",
    SOURCE_SCRIPT_DIR / "build_e10_outlier_signal_features.py",
    SOURCE_SCRIPT_DIR / "build_e09_exact_prev_features.py",
]

LOCKED_SPLIT_ORDER = ["valid", "test", "recent_holdout"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, copied_to: Path | None = None) -> dict[str, object]:
    return {
        "source_path": str(path),
        "artifact_path": str(copied_to) if copied_to else None,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_metrics(metrics_path: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    with metrics_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row["experiment_name"] == SOURCE_EXPERIMENT and row["split"] in LOCKED_SPLIT_ORDER:
                rows.append(row)
    rows.sort(key=lambda row: LOCKED_SPLIT_ORDER.index(row["split"]))
    if len(rows) != len(LOCKED_SPLIT_ORDER):
        found = [row["split"] for row in rows]
        raise RuntimeError(f"missing locked metric splits: found={found}")

    numeric_features = json.loads(rows[0]["numeric_features"])
    embedding_features = json.loads(rows[0]["embedding_features"])
    return rows, numeric_features, embedding_features


def write_final_metrics(rows: list[dict[str, str]]) -> None:
    fields = [
        "model",
        "source_experiment",
        "split",
        "rows",
        "log_mae",
        "log_rmse",
        "total_price_mae_manwon",
        "abs_pct_error_p95",
        "abs_pct_error_p99",
        "error_gt_10pct_rate",
        "error_gt_20pct_rate",
        "error_gt_30pct_rate",
        "error_gt_50pct_rate",
    ]
    with (ARTIFACT_DIR / "final_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": FINAL_MODEL,
                    "source_experiment": SOURCE_EXPERIMENT,
                    "split": row["split"],
                    "rows": row["rows"],
                    "log_mae": row["log_mae"],
                    "log_rmse": row["log_rmse"],
                    "total_price_mae_manwon": row["total_price_mae_manwon"],
                    "abs_pct_error_p95": row["abs_pct_error_p95"],
                    "abs_pct_error_p99": row["abs_pct_error_p99"],
                    "error_gt_10pct_rate": row["error_gt_10pct_rate"],
                    "error_gt_20pct_rate": row["error_gt_20pct_rate"],
                    "error_gt_30pct_rate": row["error_gt_30pct_rate"],
                    "error_gt_50pct_rate": row["error_gt_50pct_rate"],
                }
            )


def write_feature_schema(numeric_features: list[str], embedding_features: list[str]) -> None:
    schema = {
        "model": FINAL_MODEL,
        "source_experiment": SOURCE_EXPERIMENT,
        "target": "log(price_per_m2)",
        "base_log_feature": "log_complex_prev_price_per_m2",
        "numeric_features": numeric_features,
        "embedding_features": embedding_features,
        "excluded_final_candidates": [
            "regional_residual_bias_correction",
            "prev3_rolling_features",
            "monthly_market_anchor",
            "floor_feature_expansion",
            "news_region_month_price_feature",
        ],
    }
    (ARTIFACT_DIR / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: str) -> str:
    return f"{float(value) * 100:.4f}%"


def write_model_card(rows: list[dict[str, str]]) -> None:
    recent = next(row for row in rows if row["split"] == "recent_holdout")
    lines = [
        "# F18 Canonical Model Artifact",
        "",
        "## Final decision",
        f"- final price model: `{FINAL_MODEL}`",
        f"- source experiment: `{SOURCE_EXPERIMENT}`",
        "- training epoch: `30`",
        "- loss: `Huber(delta=0.10)`",
        "- price correction: not adopted",
        "- confidence/risk policy: keep separate from price prediction",
        "- trained model weights: not present in the current saved artifacts; this freeze locks the reproducible model definition and evaluation evidence.",
        "",
        "## Locked recent_holdout metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| MAE(log) | {float(recent['log_mae']):.6f} |",
        f"| p95 | {pct(recent['abs_pct_error_p95'])} |",
        f"| p99 | {pct(recent['abs_pct_error_p99'])} |",
        f"| >10% | {pct(recent['error_gt_10pct_rate'])} |",
        f"| >20% | {pct(recent['error_gt_20pct_rate'])} |",
        "",
        "## Split metrics",
        "",
        "| split | MAE(log) | p95 | p99 | >10% | >20% | rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {float(row['log_mae']):.6f} | "
            f"{pct(row['abs_pct_error_p95'])} | {pct(row['abs_pct_error_p99'])} | "
            f"{pct(row['error_gt_10pct_rate'])} | {pct(row['error_gt_20pct_rate'])} | {row['rows']} |"
        )
    lines.extend(
        [
            "",
            "## Artifact contents",
            "- `manifest.json`: locked source paths, file sizes, and SHA-256 hashes",
            "- `feature_schema.json`: final feature contract",
            "- `final_metrics.csv`: locked final model metrics",
            "- `evidence/`: copied E10 evaluation reports used for the decision",
            "",
            "## Reproduction command",
            "```bash",
            "cd /Users/gwongwangjae/goorm-ai-language-course/final_project",
            "E10_RUN_MODE=full E10_EXPERIMENTS=F18_reference_recheck,F18_reference_huber_010 E10_MAX_EPOCHS=30 python scripts/run_e10_outlier_signal_experiments.py",
            "```",
        ]
    )
    (ARTIFACT_DIR / "MODEL_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    missing = [path for path in [*EVIDENCE_FILES, *HASH_ONLY_FILES] if not path.exists()]
    if missing:
        raise SystemExit("missing source files:\n" + "\n".join(str(path) for path in missing))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows, numeric_features, embedding_features = read_metrics(SOURCE_OUTPUT_DIR / "e10_outlier_signal_metrics.csv")

    evidence_records = []
    for source in EVIDENCE_FILES:
        target = EVIDENCE_DIR / source.name
        shutil.copy2(source, target)
        evidence_records.append(file_record(source, target.relative_to(REPO_DIR)))

    hash_only_records = [file_record(path) for path in HASH_ONLY_FILES]

    write_final_metrics(metrics_rows)
    write_feature_schema(numeric_features, embedding_features)
    write_model_card(metrics_rows)

    generated_files = [
        ARTIFACT_DIR / "final_metrics.csv",
        ARTIFACT_DIR / "feature_schema.json",
        ARTIFACT_DIR / "MODEL_CARD.md",
    ]
    manifest = {
        "artifact_name": "f18_canonical_model_artifact",
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "final_model": FINAL_MODEL,
        "source_experiment": SOURCE_EXPERIMENT,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "status": "locked",
        "selection_basis": "recent_holdout log_mae primary; p95, p99, >10%, >20% retained as guardrails/evidence",
        "training_config": {
            "run_mode": "full",
            "max_epochs": 30,
            "batch_size": 8192,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "Huber(delta=0.10)",
            "dense_units": [128, 64],
            "dropout": {"unit>=128": 0.10, "else": 0.05},
            "kernel_regularizer": "l2(1e-5)",
            "seed": 42 + 183,
            "split": "train<=2023, valid=2024, test=2025, recent_holdout>=2026",
            "policy": "Policy B: is_cancelled == 0, trade_type in [중개거래, unknown]",
        },
        "copied_evidence_files": evidence_records,
        "hash_only_source_files": hash_only_records,
        "generated_artifact_files": [file_record(path, path.relative_to(REPO_DIR)) for path in generated_files],
    }
    manifest_path = ARTIFACT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "_SUCCESS").write_text("locked\n", encoding="utf-8")

    print(f"locked artifact: {ARTIFACT_DIR}")
    print(f"final model: {FINAL_MODEL}")
    print(f"recent_holdout log_mae: {next(row for row in metrics_rows if row['split'] == 'recent_holdout')['log_mae']}")


if __name__ == "__main__":
    main()
