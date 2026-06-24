#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
FINAL_PROJECT_DIR = Path("/Users/gwongwangjae/goorm-ai-language-course/final_project")
SOURCE_OUTPUT_DIR = FINAL_PROJECT_DIR / "outputs"
SOURCE_SCRIPT_DIR = FINAL_PROJECT_DIR / "scripts"

ARTIFACT_DIR = REPO_DIR / "outputs" / "f18_confidence_interval_policy"
EVIDENCE_DIR = ARTIFACT_DIR / "evidence"

FINAL_PRICE_MODEL = "canonical_F18_reference_huber_010"
POLICY_NAME = "F41_confidence_interval_policy"

CONFIDENCE_REPORT = SOURCE_OUTPUT_DIR / "e11_region_confidence_report.csv"
INTERVAL_POLICY_MD = SOURCE_OUTPUT_DIR / "e11_prediction_interval_policy.md"
E11_EVAL_PREDICTIONS = SOURCE_OUTPUT_DIR / "e11_f18_eval_predictions.csv"
POLICY_CANDIDATES = REPO_DIR / "outputs" / "f18_final_policy_sweep" / "f18_final_policy_candidates.csv"
POLICY_SWEEP_SUMMARY = REPO_DIR / "outputs" / "f18_final_policy_sweep" / "f18_final_policy_sweep_summary.md"
POLICY_SWEEP_DECISION = REPO_DIR / "outputs" / "f18_final_policy_sweep" / "f18_final_policy_sweep_final_decision.md"

COPIED_EVIDENCE_FILES = [
    INTERVAL_POLICY_MD,
    POLICY_CANDIDATES,
    POLICY_SWEEP_SUMMARY,
    POLICY_SWEEP_DECISION,
]

HASH_ONLY_FILES = [
    CONFIDENCE_REPORT,
    E11_EVAL_PREDICTIONS,
    SOURCE_SCRIPT_DIR / "build_e11_region_residual_features.py",
    SOURCE_SCRIPT_DIR / "run_e11_region_residual_experiments.py",
    REPO_DIR / "scripts" / "run_f18_final_policy_sweep.py",
]

FALLBACK_ORDER = ["complex", "legal_dong", "sgg", "sido", "global"]


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


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def load_policy_candidate() -> dict[str, str]:
    with POLICY_CANDIDATES.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row["candidate"] == POLICY_NAME:
                return row
    raise RuntimeError(f"{POLICY_NAME} not found in {POLICY_CANDIDATES}")


def summarize_confidence_report() -> dict[str, object]:
    level_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    level_tier_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tier_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_row: dict[str, str] | None = None

    with CONFIDENCE_REPORT.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            level = row["level"]
            tier = row["confidence_tier"]
            level_counts[level] += 1
            tier_counts[tier] += 1
            level_tier_counts[level][tier] += 1
            for col in ["p80_abs_pct_error", "p90_abs_pct_error", "p95_abs_pct_error"]:
                tier_values[tier][col].append(float(row[col]))
            if level == "global":
                global_row = row

    if global_row is None:
        raise RuntimeError(f"global row missing in {CONFIDENCE_REPORT}")

    tier_summary = []
    for tier in sorted(tier_counts):
        tier_summary.append(
            {
                "confidence_tier": tier,
                "groups": tier_counts[tier],
                "median_p80": median(tier_values[tier]["p80_abs_pct_error"]),
                "median_p90": median(tier_values[tier]["p90_abs_pct_error"]),
                "median_p95": median(tier_values[tier]["p95_abs_pct_error"]),
            }
        )

    return {
        "rows": sum(level_counts.values()),
        "level_counts": dict(level_counts),
        "tier_counts": dict(tier_counts),
        "level_tier_counts": {level: dict(counter) for level, counter in level_tier_counts.items()},
        "global": {
            "source_until_ym": global_row["source_until_ym"],
            "rows": int(global_row["rows"]),
            "p80_abs_pct_error": float(global_row["p80_abs_pct_error"]),
            "p90_abs_pct_error": float(global_row["p90_abs_pct_error"]),
            "p95_abs_pct_error": float(global_row["p95_abs_pct_error"]),
            "confidence_tier": global_row["confidence_tier"],
        },
        "tier_summary": tier_summary,
    }


def compute_eval_lift() -> dict[str, object]:
    total = 0
    gt20 = 0
    by_tier: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "gt20": 0})
    with E11_EVAL_PREDICTIONS.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row["split"] != "recent_holdout":
                continue
            tier = row["resid_risk_tier"] or "unknown"
            abs_pct = float(row["raw_f18_abs_pct_error"])
            total += 1
            is_gt20 = abs_pct > 0.20
            gt20 += int(is_gt20)
            by_tier[tier]["rows"] += 1
            by_tier[tier]["gt20"] += int(is_gt20)

    if total == 0:
        raise RuntimeError(f"recent_holdout rows missing in {E11_EVAL_PREDICTIONS}")

    overall_gt20_rate = gt20 / total
    tier_rows = []
    for tier in sorted(by_tier):
        rows = by_tier[tier]["rows"]
        rate = by_tier[tier]["gt20"] / rows if rows else 0.0
        tier_rows.append(
            {
                "resid_risk_tier": tier,
                "rows": rows,
                "coverage": rows / total,
                "error_gt_20pct_rate": rate,
                "lift_vs_overall": rate / overall_gt20_rate if overall_gt20_rate else 0.0,
            }
        )
    high = next((row for row in tier_rows if row["resid_risk_tier"] == "high"), None)
    return {
        "recent_holdout_rows": total,
        "overall_error_gt_20pct_rate": overall_gt20_rate,
        "tier_rows": tier_rows,
        "high_risk": high,
    }


def write_risk_metrics_csv(candidate: dict[str, str], eval_lift: dict[str, object]) -> None:
    path = ARTIFACT_DIR / "risk_policy_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "policy",
                "basis",
                "status",
                "flag_definition",
                "flagged_rows",
                "flagged_coverage",
                "flagged_error_gt_20pct_rate",
                "overall_error_gt_20pct_rate",
                "lift_vs_overall",
            ],
        )
        writer.writeheader()
        high = eval_lift["high_risk"] or {}
        writer.writerow(
            {
                "policy": POLICY_NAME,
                "basis": candidate["basis"],
                "status": "adopted_risk_policy",
                "flag_definition": "resid_risk_tier == high",
                "flagged_rows": high.get("rows", candidate["rows"]),
                "flagged_coverage": high.get("coverage", ""),
                "flagged_error_gt_20pct_rate": high.get("error_gt_20pct_rate", candidate["gt20"]),
                "overall_error_gt_20pct_rate": eval_lift["overall_error_gt_20pct_rate"],
                "lift_vs_overall": high.get("lift_vs_overall", candidate["lift_or_delta"]),
            }
        )


def write_policy_json(conf_summary: dict[str, object], eval_lift: dict[str, object]) -> None:
    policy = {
        "policy_name": POLICY_NAME,
        "status": "adopted_as_risk_policy",
        "price_model": FINAL_PRICE_MODEL,
        "price_prediction_changes": False,
        "purpose": "Expose confidence tier and expected percent error interval without changing the price prediction.",
        "lookup_report": {
            "path": str(CONFIDENCE_REPORT),
            "source_until_ym": conf_summary["global"]["source_until_ym"],
            "fallback_order": FALLBACK_ORDER,
            "required_lookup_keys": {
                "complex": "complex_id",
                "legal_dong": "legal_dong_code",
                "sgg": "sgg_code",
                "sido": "sido_code",
                "global": "global",
            },
        },
        "interval_rule": {
            "default_interval_pct": "p90_abs_pct_error",
            "conservative_interval_pct": "p95_abs_pct_error",
            "price_low": "predicted_price * max(0, 1 - interval_pct)",
            "price_high": "predicted_price * (1 + interval_pct)",
        },
        "confidence_tier_rule": {
            "high": "rows >= 300 and p90_abs_pct_error <= 0.20",
            "medium": "rows >= 50 and p90_abs_pct_error <= 0.30",
            "low": "rows > 0",
            "no_history": "no residual history; use global fallback",
        },
        "risk_warning_rule": {
            "warn": "confidence_tier == low or resid_risk_tier in [high, unknown]",
            "primary_evaluated_flag": "resid_risk_tier == high",
        },
        "evaluation": eval_lift,
        "confidence_report_summary": conf_summary,
    }
    (ARTIFACT_DIR / "confidence_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_policy_card(candidate: dict[str, str], conf_summary: dict[str, object], eval_lift: dict[str, object]) -> None:
    high = eval_lift["high_risk"] or {}
    global_row = conf_summary["global"]
    lines = [
        "# F18 Confidence Interval Policy",
        "",
        "## Final decision",
        f"- policy: `{POLICY_NAME}`",
        "- status: `adopted_as_risk_policy`",
        f"- price model: `{FINAL_PRICE_MODEL}`",
        "- price prediction: unchanged",
        "- purpose: show expected error interval and confidence/risk warning.",
        "",
        "## What this policy does",
        "- It does not improve MAE/p95/p99 directly.",
        "- It uses historical residual error by complex/region to show an expected percent error range.",
        "- It prioritizes user caution when the model is likely unstable.",
        "",
        "## Evaluation signal",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| recent_holdout rows | {eval_lift['recent_holdout_rows']} |",
        f"| high-risk rows | {high.get('rows', 0)} |",
        f"| high-risk coverage | {pct(float(high.get('coverage', 0.0)))} |",
        f"| high-risk >20% error rate | {pct(float(high.get('error_gt_20pct_rate', 0.0)))} |",
        f"| overall >20% error rate | {pct(float(eval_lift['overall_error_gt_20pct_rate']))} |",
        f"| lift vs overall | {float(high.get('lift_vs_overall', 0.0)):.2f}x |",
        "",
        "## Interval lookup rule",
        "",
        "| priority | level | key |",
        "| ---: | --- | --- |",
        "| 1 | complex | `complex_id` |",
        "| 2 | legal_dong | `legal_dong_code` |",
        "| 3 | sgg | `sgg_code` |",
        "| 4 | sido | `sido_code` |",
        "| 5 | global | fallback |",
        "",
        "Use `p90_abs_pct_error` as the default interval and `p95_abs_pct_error` when a conservative range is needed.",
        "",
        "```text",
        "lower = predicted_price * max(0, 1 - interval_pct)",
        "upper = predicted_price * (1 + interval_pct)",
        "```",
        "",
        "## Confidence tier rule",
        "",
        "| tier | rule |",
        "| --- | --- |",
        "| high | rows >= 300 and p90 <= 20% |",
        "| medium | rows >= 50 and p90 <= 30% |",
        "| low | rows > 0 but not high/medium |",
        "| no_history | no residual history; use fallback |",
        "",
        "## Global fallback",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| source_until_ym | {global_row['source_until_ym']} |",
        f"| rows | {global_row['rows']} |",
        f"| p80 | {pct(float(global_row['p80_abs_pct_error']))} |",
        f"| p90 | {pct(float(global_row['p90_abs_pct_error']))} |",
        f"| p95 | {pct(float(global_row['p95_abs_pct_error']))} |",
        "",
        "## Tier summary",
        "",
        "| confidence_tier | groups | median p80 | median p90 | median p95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in conf_summary["tier_summary"]:
        lines.append(
            f"| {row['confidence_tier']} | {row['groups']} | {pct(row['median_p80'])} | {pct(row['median_p90'])} | {pct(row['median_p95'])} |"
        )
    lines.extend(
        [
            "",
            "## Display policy",
            "- normal: show price plus `예상 오차 범위 ±p90`.",
            "- low confidence or high/unknown residual risk: show price as reference and emphasize confidence warning.",
            "- do not apply this as a correction to the predicted price.",
            "",
            "## Source note",
            f"- candidate reason: `{candidate['reason']}`",
            "- F41 lift is measured on E11 same-run eval predictions, not as a replacement for the canonical F18 price metric.",
        ]
    )
    (ARTIFACT_DIR / "POLICY_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    missing = [path for path in [*COPIED_EVIDENCE_FILES, *HASH_ONLY_FILES] if not path.exists()]
    if missing:
        raise SystemExit("missing source files:\n" + "\n".join(str(path) for path in missing))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    candidate = load_policy_candidate()
    conf_summary = summarize_confidence_report()
    eval_lift = compute_eval_lift()

    copied_records = []
    for source in COPIED_EVIDENCE_FILES:
        target = EVIDENCE_DIR / source.name
        shutil.copy2(source, target)
        copied_records.append(file_record(source, target.relative_to(REPO_DIR)))

    write_risk_metrics_csv(candidate, eval_lift)
    write_policy_json(conf_summary, eval_lift)
    write_policy_card(candidate, conf_summary, eval_lift)

    generated_files = [
        ARTIFACT_DIR / "risk_policy_metrics.csv",
        ARTIFACT_DIR / "confidence_policy.json",
        ARTIFACT_DIR / "POLICY_CARD.md",
    ]
    manifest = {
        "artifact_name": "f18_confidence_interval_policy",
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "policy": POLICY_NAME,
        "status": "adopted_as_risk_policy",
        "price_model": FINAL_PRICE_MODEL,
        "copied_evidence_files": copied_records,
        "hash_only_source_files": [file_record(path) for path in HASH_ONLY_FILES],
        "generated_artifact_files": [file_record(path, path.relative_to(REPO_DIR)) for path in generated_files],
    }
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "_SUCCESS").write_text("locked\n", encoding="utf-8")

    high = eval_lift["high_risk"] or {}
    print(f"locked policy: {ARTIFACT_DIR}")
    print(f"policy: {POLICY_NAME}")
    print(f"high risk coverage: {float(high.get('coverage', 0.0)):.4f}")
    print(f"high risk gt20: {float(high.get('error_gt_20pct_rate', 0.0)):.4f}")
    print(f"lift: {float(high.get('lift_vs_overall', 0.0)):.2f}")


if __name__ == "__main__":
    main()
