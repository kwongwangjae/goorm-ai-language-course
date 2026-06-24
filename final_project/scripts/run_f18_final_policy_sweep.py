#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path("/Users/gwongwangjae/goorm-ai-language-course/final_project")
OUTPUT_DIR = REPO_ROOT / "outputs" / "f18_final_policy_sweep"
METRICS_CSV = OUTPUT_DIR / "f18_final_policy_candidates.csv"
SUMMARY_MD = OUTPUT_DIR / "f18_final_policy_sweep_summary.md"
FINAL_DECISION_MD = OUTPUT_DIR / "f18_final_policy_sweep_final_decision.md"
SUCCESS_PATH = OUTPUT_DIR / "_SUCCESS"

E11_EVAL = PROJECT_DIR / "outputs" / "e11_f18_eval_predictions.csv"
TRANSACTIONS = PROJECT_DIR / "data" / "processed" / "transactions.csv"
EXACT_PREV = PROJECT_DIR / "outputs" / "e09_exact_prev_features.csv"
E10_METRICS = PROJECT_DIR / "outputs" / "e10_outlier_signal_metrics.csv"
MONTHLY_SUMMARY = REPO_ROOT / "outputs" / "f18_monthly_anchor_tuning" / "f18_monthly_anchor_tuning_summary.md"
MONTHLY_GROUP_METRICS = REPO_ROOT / "outputs" / "f18_monthly_anchor_tuning" / "f18_monthly_anchor_tuning_group_metrics.csv"
NEWS_SIDECAR = REPO_ROOT / "outputs" / "f18_news_tuning" / "f18_news_region_month_features.csv"

CANONICAL = {
    "candidate": "canonical_F18_reference_huber_010",
    "log_mae": 0.06177453101105649,
    "p95": 0.18807662850545506,
    "p99": 0.34558166944175545,
    "gt10": 0.17747340882616916,
    "gt20": 0.04295166803521302,
}


@dataclass(frozen=True)
class CandidateResult:
    order: int
    candidate: str
    kind: str
    status: str
    basis: str
    rows: int | None
    log_mae: float | None
    p95: float | None
    p99: float | None
    gt10: float | None
    gt20: float | None
    lift_or_delta: float | None
    reason: str


def metric_from_pred(df: pd.DataFrame, pred_col: str, candidate: str) -> dict[str, Any]:
    pred = df[pred_col].to_numpy(dtype="float64")
    actual = df["actual_price_per_m2"].to_numpy(dtype="float64")
    abs_pct = np.abs((pred - actual) / actual)
    log_mae = np.abs(np.log(pred) - np.log(actual)).mean()
    return {
        "candidate": candidate,
        "rows": len(df),
        "log_mae": float(log_mae),
        "p95": float(np.quantile(abs_pct, 0.95)),
        "p99": float(np.quantile(abs_pct, 0.99)),
        "gt10": float((abs_pct > 0.10).mean()),
        "gt20": float((abs_pct > 0.20).mean()),
    }


def load_eval_frame() -> pd.DataFrame:
    eval_df = pd.read_csv(E11_EVAL, dtype={"transaction_id": "string"})
    eval_df = eval_df.loc[eval_df["split"].isin(["valid", "test", "recent_holdout"])].copy()
    raw = pd.read_csv(
        TRANSACTIONS,
        usecols=["transaction_id", "prev_deal_gap_days", "complex_prev_missing"],
        dtype={"transaction_id": "string", "prev_deal_gap_days": "float32", "complex_prev_missing": "Int8"},
    )
    exact = pd.read_csv(
        EXACT_PREV,
        usecols=["transaction_id", "exact_prev1_missing", "exact_prev1_gap_days", "wide_prev1_present_exact_missing"],
        dtype={
            "transaction_id": "string",
            "exact_prev1_missing": "Int8",
            "exact_prev1_gap_days": "float32",
            "wide_prev1_present_exact_missing": "Int8",
        },
    )
    return eval_df.merge(raw, on="transaction_id", how="left", validate="one_to_one").merge(
        exact, on="transaction_id", how="left", validate="one_to_one"
    )


def f40_sparse_fallback(eval_df: pd.DataFrame) -> CandidateResult:
    recent = eval_df.loc[eval_df["split"] == "recent_holdout"].copy()
    ref = metric_from_pred(recent, "raw_f18_pred_price_per_m2", "same_run_F18_raw")
    sparse = (
        recent["exact_prev1_missing"].fillna(1).astype("int8").eq(1)
        | recent["wide_prev1_present_exact_missing"].fillna(0).astype("int8").eq(1)
        | pd.to_numeric(recent["exact_prev1_gap_days"], errors="coerce").fillna(9999).gt(365)
        | pd.to_numeric(recent["prev_deal_gap_days"], errors="coerce").fillna(9999).gt(365)
    )
    recent["f40_pred"] = np.where(sparse, recent["feature_model_pred_price_per_m2"], recent["raw_f18_pred_price_per_m2"])
    got = metric_from_pred(recent, "f40_pred", "F40_sparse_fallback_policy")
    d_mae = got["log_mae"] - ref["log_mae"]
    d_p99 = got["p99"] - ref["p99"]
    d_gt20 = got["gt20"] - ref["gt20"]
    beats_same_run = d_mae < -0.0001 and (d_p99 <= 0.003 or d_gt20 <= 0.001)
    beats_canonical = got["log_mae"] < CANONICAL["log_mae"] and got["p99"] <= CANONICAL["p99"] + 0.003 and got["gt20"] <= CANONICAL["gt20"] + 0.001
    status = "discarded"
    reason = (
        f"sparse rows={int(sparse.sum())}; same-run d_MAE={d_mae:.6f}, d_p99={d_p99:.6f}, d_gt20={d_gt20:.6f}; "
        f"beats_same_run={beats_same_run}, beats_canonical={beats_canonical}"
    )
    if beats_same_run and beats_canonical:
        status = "candidate"
    return CandidateResult(1, "F40_sparse_fallback_policy", "price_policy", status, "row-level E11 F18/F29 predictions", got["rows"], got["log_mae"], got["p95"], got["p99"], got["gt10"], got["gt20"], d_mae, reason)


def f41_confidence_interval(eval_df: pd.DataFrame) -> CandidateResult:
    recent = eval_df.loc[eval_df["split"] == "recent_holdout"].copy()
    overall_gt20 = float((recent["raw_f18_abs_pct_error"] > 0.20).mean())
    high = recent.loc[recent["resid_risk_tier"].eq("high")]
    high_gt20 = float((high["raw_f18_abs_pct_error"] > 0.20).mean()) if len(high) else 0.0
    lift = high_gt20 / overall_gt20 if overall_gt20 else 0.0
    status = "candidate" if len(high) / len(recent) >= 0.05 and lift >= 1.5 else "discarded"
    reason = f"high-risk coverage={len(high) / len(recent):.4f}; high-risk >20%={high_gt20:.4f}; overall >20%={overall_gt20:.4f}; lift={lift:.2f}"
    return CandidateResult(2, "F41_confidence_interval_policy", "risk_policy", status, "E11 residual risk tier and interval policy", len(high), None, None, None, None, high_gt20, lift, reason)


def f42_monthly_anchor_gated() -> CandidateResult:
    group_df = pd.read_csv(MONTHLY_GROUP_METRICS)
    recent = group_df.loc[group_df["split"].eq("recent_holdout")].copy()
    ref = recent.loc[recent["experiment_name"].eq("F18_reference_huber_010")]
    monthly = recent.loc[recent["experiment_name"].eq("F36_monthly_market_anchor_huber_010")]
    checks = []
    for group_type, group_value in [
        ("exact_prev1_gap_bucket_plus", "366-730"),
        ("exact_prev1_gap_bucket_plus", "731+"),
        ("prev1_gap_bucket_plus", "366-730"),
        ("prev1_gap_bucket_plus", "731+"),
        ("exact_prev1_missing_group", "1"),
    ]:
        a = ref.loc[(ref["group_type"].eq(group_type)) & (ref["group_value"].astype(str).eq(group_value))]
        b = monthly.loc[(monthly["group_type"].eq(group_type)) & (monthly["group_value"].astype(str).eq(group_value))]
        if a.empty or b.empty:
            continue
        a = a.iloc[0]
        b = b.iloc[0]
        checks.append(
            {
                "group": f"{group_type}={group_value}",
                "d_mae": float(b["log_mae"] - a["log_mae"]),
                "d_p99": float(b["p99_abs_pct_error"] - a["p99_abs_pct_error"]),
                "d_gt20": float(b["error_gt_20pct_rate"] - a["error_gt_20pct_rate"]),
            }
        )
    p99_worse = [c for c in checks if c["d_p99"] > 0.003]
    mae_better = [c for c in checks if c["d_mae"] < -0.0001]
    # Full monthly anchor still misses canonical F18, and long-gap p99 worsens in key buckets.
    status = "discarded"
    reason = f"long-gap groups with MAE gain={len(mae_better)}/{len(checks)}, p99 worse={len(p99_worse)}/{len(checks)}; full monthly did not beat canonical"
    return CandidateResult(3, "F42_monthly_anchor_gated", "price_policy", status, "monthly anchor group metrics", None, None, None, None, None, None, None, reason)


def f43_reconstruction_news_risk(eval_df: pd.DataFrame) -> CandidateResult:
    if not NEWS_SIDECAR.exists():
        return CandidateResult(4, "F43_reconstruction_news_risk_only", "risk_policy", "discarded", "news sidecar missing", None, None, None, None, None, None, None, "news sidecar missing")
    usecols = [
        "transaction_id",
        "detail_lag1m_redevelopment_score",
        "detail_lag3m_redevelopment_score",
        "detail_lag1m_transport_score",
        "detail_lag3m_transport_score",
        "detail_lag1m_supply_risk_score",
        "detail_lag3m_supply_risk_score",
        "detail_lag1m_price_net_signal",
        "detail_lag3m_price_net_signal",
        "province_lag1m_redevelopment_score",
        "province_lag3m_redevelopment_score",
        "province_lag1m_price_net_signal",
        "province_lag3m_price_net_signal",
    ]
    news = pd.read_csv(NEWS_SIDECAR, usecols=usecols, dtype={"transaction_id": "string"})
    df = eval_df.loc[eval_df["split"].isin(["valid", "recent_holdout"]), ["transaction_id", "split", "raw_f18_abs_pct_error"]].merge(
        news, on="transaction_id", how="left", validate="one_to_one"
    )
    score_cols = [col for col in usecols if col != "transaction_id"]
    df[score_cols] = df[score_cols].fillna(0)
    df["news_risk_score"] = (
        df[["detail_lag1m_redevelopment_score", "detail_lag3m_redevelopment_score", "province_lag1m_redevelopment_score", "province_lag3m_redevelopment_score"]].max(axis=1)
        + 0.5 * df[["detail_lag1m_transport_score", "detail_lag3m_transport_score", "detail_lag1m_supply_risk_score", "detail_lag3m_supply_risk_score"]].max(axis=1)
        + 0.25 * df[["detail_lag1m_price_net_signal", "detail_lag3m_price_net_signal", "province_lag1m_price_net_signal", "province_lag3m_price_net_signal"]].abs().max(axis=1)
    )
    valid = df.loc[df["split"].eq("valid")]
    threshold = float(valid["news_risk_score"].quantile(0.90))
    recent = df.loc[df["split"].eq("recent_holdout")].copy()
    flagged = recent.loc[recent["news_risk_score"].ge(threshold)]
    overall_gt20 = float((recent["raw_f18_abs_pct_error"] > 0.20).mean())
    flagged_gt20 = float((flagged["raw_f18_abs_pct_error"] > 0.20).mean()) if len(flagged) else 0.0
    lift = flagged_gt20 / overall_gt20 if overall_gt20 else 0.0
    status = "candidate" if len(flagged) / len(recent) >= 0.03 and lift >= 1.3 else "discarded"
    reason = f"threshold={threshold:.4f}; flagged coverage={len(flagged) / len(recent):.4f}; flagged >20%={flagged_gt20:.4f}; overall >20%={overall_gt20:.4f}; lift={lift:.2f}"
    return CandidateResult(4, "F43_reconstruction_news_risk_only", "risk_policy", status, "news/reconstruction risk flag lift", len(flagged), None, None, None, None, flagged_gt20, lift, reason)


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value * 100:.4f}%"


def num(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.6f}"


def write_outputs(results: list[CandidateResult]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["order", "candidate", "kind", "status", "basis", "rows", "log_mae", "p95", "p99", "gt10", "gt20", "lift_or_delta", "reason"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
    lines = [
        "# F18 Final Policy Sweep Summary",
        "",
        "## Canonical Reference",
        "",
        "| candidate | MAE | p95 | p99 | >10% | >20% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| {CANONICAL['candidate']} | {num(CANONICAL['log_mae'])} | {pct(CANONICAL['p95'])} | {pct(CANONICAL['p99'])} | {pct(CANONICAL['gt10'])} | {pct(CANONICAL['gt20'])} |",
        "",
        "## Candidate Results",
        "",
        "| order | candidate | kind | status | MAE | p95 | p99 | >10% | >20% / risk >20% | lift/delta | reason |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.order} | {r.candidate} | {r.kind} | {r.status} | {num(r.log_mae)} | {pct(r.p95)} | {pct(r.p99)} | {pct(r.gt10)} | {pct(r.gt20)} | {num(r.lift_or_delta)} | {r.reason} |"
        )
    adopted = [r for r in results if r.status == "candidate" and r.kind == "risk_policy"]
    lines.extend(
        [
            "",
            "## Final",
            "",
            "- price model: `canonical_F18_reference_huber_010`",
            f"- adopted risk policies: `{', '.join(r.candidate for r in adopted) if adopted else 'none'}`",
            "- discarded price policies: `F40_sparse_fallback_policy`, `F42_monthly_anchor_gated`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    FINAL_DECISION_MD.write_text(
        "\n".join(
            [
                "# F18 Final Policy Sweep Decision",
                "",
                "- final price model: `canonical_F18_reference_huber_010`",
                f"- risk policy candidate(s): `{', '.join(r.candidate for r in adopted) if adopted else 'none'}`",
                "- F40/F42 are not adopted as price policies.",
                "- F43 is adopted only if it shows useful lift; otherwise reconstruction/news remains explanatory only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    SUCCESS_PATH.write_text(f"completed_at_utc={pd.Timestamp.utcnow().isoformat()}\n", encoding="utf-8")


def main() -> int:
    eval_df = load_eval_frame()
    results = [
        f40_sparse_fallback(eval_df),
        f41_confidence_interval(eval_df),
        f42_monthly_anchor_gated(),
        f43_reconstruction_news_risk(eval_df),
    ]
    write_outputs(results)
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
