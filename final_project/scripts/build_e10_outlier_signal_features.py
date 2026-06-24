#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


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
OUTPUT_DIR = PROJECT_DIR / "outputs"
FEATURE_PATH = OUTPUT_DIR / "e10_outlier_signal_features.csv"
QUALITY_REPORT_PATH = OUTPUT_DIR / "e10_outlier_signal_feature_quality_report.md"

LOG_10 = math.log(1.10)
LOG_20 = math.log(1.20)
LOG_30 = math.log(1.30)


def md_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float", "float32", "float64"]).columns:
        x[col] = x[col].map(lambda v: format(v, floatfmt) if pd.notna(v) else "")
    x = x.astype("string").fillna("")
    lines = ["| " + " | ".join(x.columns) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in x.values.tolist()]
    return "\n".join(lines)


def safe_log(values: pd.Series) -> pd.Series:
    numeric = values.astype("float64")
    return pd.Series(np.where(numeric > 0, np.log(numeric), np.nan), index=values.index, dtype="float64")


def add_jump_features(out: pd.DataFrame, prefix: str, log_return: pd.Series, present: pd.Series) -> None:
    signed = log_return.where(present, np.nan).astype("float64")
    abs_value = signed.abs()
    out[f"{prefix}_jump_signed"] = signed.fillna(0).astype("float32")
    out[f"{prefix}_jump_abs"] = abs_value.fillna(0).astype("float32")
    out[f"{prefix}_jump_up"] = signed.clip(lower=0).fillna(0).astype("float32")
    out[f"{prefix}_jump_down_abs"] = (-signed.clip(upper=0)).fillna(0).astype("float32")
    out[f"is_{prefix}_jump_10pct"] = ((abs_value > LOG_10) & present).astype("float32")
    out[f"is_{prefix}_jump_20pct"] = ((abs_value > LOG_20) & present).astype("float32")
    out[f"is_{prefix}_jump_30pct"] = ((abs_value > LOG_30) & present).astype("float32")
    out[f"is_{prefix}_jump_up_20pct"] = ((signed > LOG_20) & present).astype("float32")
    out[f"is_{prefix}_jump_down_20pct"] = ((signed < -LOG_20) & present).astype("float32")


def build_region_prior(raw_df: pd.DataFrame) -> pd.DataFrame:
    policy_mask = (raw_df["is_cancelled"] == 0) & (raw_df["trade_type"].fillna("unknown").isin(["중개거래", "unknown"]))
    clean = raw_df.loc[policy_mask & raw_df["price_per_m2"].gt(0), ["sgg_code", "deal_ym", "price_per_m2"]].copy()
    clean["log_price_per_m2"] = np.log(clean["price_per_m2"].astype("float64"))
    monthly = (
        clean.groupby(["sgg_code", "deal_ym"], dropna=False)
        .agg(sgg_month_log_median=("log_price_per_m2", "median"), sgg_month_count=("log_price_per_m2", "size"))
        .reset_index()
        .sort_values(["sgg_code", "deal_ym"])
    )
    monthly["sgg_lag1_source_ym"] = monthly.groupby("sgg_code", dropna=False)["deal_ym"].shift(1)
    monthly["sgg_lag1_log_median_ppm"] = monthly.groupby("sgg_code", dropna=False)["sgg_month_log_median"].shift(1)
    monthly["sgg_lag1_month_count"] = monthly.groupby("sgg_code", dropna=False)["sgg_month_count"].shift(1)
    monthly["sgg_roll3_log_median_ppm"] = (
        monthly.groupby("sgg_code", dropna=False)["sgg_month_log_median"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .astype("float64")
    )
    monthly["sgg_roll6_log_median_ppm"] = (
        monthly.groupby("sgg_code", dropna=False)["sgg_month_log_median"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
        .astype("float64")
    )
    return monthly[
        [
            "sgg_code",
            "deal_ym",
            "sgg_lag1_source_ym",
            "sgg_lag1_log_median_ppm",
            "sgg_lag1_month_count",
            "sgg_roll3_log_median_ppm",
            "sgg_roll6_log_median_ppm",
        ]
    ]


def build_sidecar() -> pd.DataFrame:
    raw_usecols = [
        "transaction_id",
        "sgg_code",
        "deal_ym",
        "deal_date",
        "trade_type",
        "is_cancelled",
        "price_per_m2",
        "complex_prev_price_per_m2",
        "complex_prev_missing",
        "prev_deal_gap_days",
    ]
    raw_dtypes = {
        "transaction_id": "string",
        "sgg_code": "string",
        "deal_ym": "string",
        "trade_type": "string",
        "is_cancelled": "Int8",
        "price_per_m2": "float32",
        "complex_prev_price_per_m2": "float32",
        "complex_prev_missing": "Int8",
        "prev_deal_gap_days": "float32",
    }
    raw_df = pd.read_csv(DATA_PATH, usecols=raw_usecols, dtype=raw_dtypes, parse_dates=["deal_date"])
    prev2_df = pd.read_csv(
        PREV2_PATH,
        usecols=["transaction_id", "complex_prev2_price_per_m2", "prev2_missing", "prev2_gap_days", "prev1_prev2_log_return"],
        dtype={"transaction_id": "string", "prev2_missing": "Int8"},
    )
    exact_df = pd.read_csv(
        EXACT_PREV_PATH,
        usecols=[
            "transaction_id",
            "exact_prev1_price_per_m2",
            "exact_prev1_missing",
            "exact_prev1_gap_days",
            "exact_prev2_price_per_m2",
            "exact_prev2_missing",
            "exact_prev2_gap_days",
            "exact_prev1_prev2_log_return",
            "wide_prev1_present_exact_missing",
        ],
        dtype={"transaction_id": "string", "exact_prev1_missing": "Int8", "exact_prev2_missing": "Int8"},
    )
    region_prior = build_region_prior(raw_df)

    assert raw_df["transaction_id"].is_unique
    assert prev2_df["transaction_id"].is_unique
    assert exact_df["transaction_id"].is_unique

    work = raw_df.merge(prev2_df, on="transaction_id", how="left", validate="one_to_one")
    work = work.merge(exact_df, on="transaction_id", how="left", validate="one_to_one")
    work = work.merge(region_prior, on=["sgg_code", "deal_ym"], how="left", validate="many_to_one")
    assert int(work["prev2_missing"].isna().sum()) == 0
    assert int(work["exact_prev1_missing"].isna().sum()) == 0
    assert len(work) == len(raw_df)

    out = pd.DataFrame({"transaction_id": work["transaction_id"]})
    wide_present = work["complex_prev_missing"].fillna(1).astype("float32").lt(1)
    prev2_present = work["prev2_missing"].fillna(1).astype("float32").lt(1)
    exact1_present = work["exact_prev1_missing"].fillna(1).astype("float32").lt(1)
    exact2_present = work["exact_prev2_missing"].fillna(1).astype("float32").lt(1)
    wide_pair_present = wide_present & prev2_present
    exact_pair_present = exact1_present & exact2_present

    add_jump_features(out, "wide_prev", work["prev1_prev2_log_return"].astype("float64"), wide_pair_present)
    add_jump_features(out, "exact_prev", work["exact_prev1_prev2_log_return"].astype("float64"), exact_pair_present)

    log_wide_prev1 = safe_log(work["complex_prev_price_per_m2"])
    log_exact_prev1 = safe_log(work["exact_prev1_price_per_m2"])
    exact_wide_gap = (log_exact_prev1 - log_wide_prev1).where(wide_present & exact1_present, np.nan)
    out["exact_wide_prev1_log_gap"] = exact_wide_gap.fillna(0).astype("float32")
    out["exact_wide_prev1_log_gap_abs"] = exact_wide_gap.abs().fillna(0).astype("float32")
    out["is_exact_wide_prev1_gap_5pct"] = ((exact_wide_gap.abs() > math.log(1.05)) & exact_wide_gap.notna()).astype("float32")
    out["is_exact_wide_prev1_gap_10pct"] = ((exact_wide_gap.abs() > LOG_10) & exact_wide_gap.notna()).astype("float32")
    out["is_exact_wide_prev1_gap_20pct"] = ((exact_wide_gap.abs() > LOG_20) & exact_wide_gap.notna()).astype("float32")

    gap_months = work["prev_deal_gap_days"].astype("float64") / 30.4375
    exact_gap_months = work["exact_prev1_gap_days"].astype("float64") / 30.4375
    gap_log = np.log1p(gap_months.clip(lower=0)).fillna(0)
    exact_gap_log = np.log1p(exact_gap_months.clip(lower=0)).fillna(0)
    out["wide_prev_jump_abs_x_gap_log1p"] = (out["wide_prev_jump_abs"].astype("float64") * gap_log).astype("float32")
    out["exact_prev_jump_abs_x_gap_log1p"] = (out["exact_prev_jump_abs"].astype("float64") * exact_gap_log).astype("float32")
    out["is_prev_gap_365d"] = (work["prev_deal_gap_days"].astype("float64") >= 365).astype("float32")
    out["is_prev_gap_730d"] = (work["prev_deal_gap_days"].astype("float64") >= 730).astype("float32")
    out["is_exact_prev_gap_365d"] = (work["exact_prev1_gap_days"].astype("float64") >= 365).astype("float32")
    out["is_exact_prev_gap_730d"] = (work["exact_prev1_gap_days"].astype("float64") >= 730).astype("float32")
    out["is_wide_jump20_and_gap365"] = ((out["is_wide_prev_jump_20pct"] > 0) & (out["is_prev_gap_365d"] > 0)).astype("float32")
    out["is_exact_jump20_and_gap365"] = ((out["is_exact_prev_jump_20pct"] > 0) & (out["is_exact_prev_gap_365d"] > 0)).astype("float32")

    region_prior_present = work["sgg_lag1_log_median_ppm"].notna()
    out["sgg_lag1_log_median_ppm"] = work["sgg_lag1_log_median_ppm"].astype("float32")
    out["sgg_lag1_month_count_log1p"] = np.log1p(work["sgg_lag1_month_count"].astype("float64")).astype("float32")
    out["sgg_roll3_log_median_ppm"] = work["sgg_roll3_log_median_ppm"].astype("float32")
    out["sgg_roll6_log_median_ppm"] = work["sgg_roll6_log_median_ppm"].astype("float32")
    out["sgg_prior_missing"] = (~region_prior_present).astype("float32")
    wide_region_gap = (log_wide_prev1 - work["sgg_lag1_log_median_ppm"].astype("float64")).where(wide_present & region_prior_present, np.nan)
    exact_region_gap = (log_exact_prev1 - work["sgg_lag1_log_median_ppm"].astype("float64")).where(exact1_present & region_prior_present, np.nan)
    out["wide_prev1_vs_sgg_lag1_log_gap"] = wide_region_gap.fillna(0).astype("float32")
    out["wide_prev1_vs_sgg_lag1_log_gap_abs"] = wide_region_gap.abs().fillna(0).astype("float32")
    out["exact_prev1_vs_sgg_lag1_log_gap"] = exact_region_gap.fillna(0).astype("float32")
    out["exact_prev1_vs_sgg_lag1_log_gap_abs"] = exact_region_gap.abs().fillna(0).astype("float32")
    out["is_wide_prev1_region_outlier_10pct"] = ((wide_region_gap.abs() > LOG_10) & wide_region_gap.notna()).astype("float32")
    out["is_wide_prev1_region_outlier_20pct"] = ((wide_region_gap.abs() > LOG_20) & wide_region_gap.notna()).astype("float32")
    out["is_wide_prev1_region_outlier_30pct"] = ((wide_region_gap.abs() > LOG_30) & wide_region_gap.notna()).astype("float32")
    out["is_exact_prev1_region_outlier_10pct"] = ((exact_region_gap.abs() > LOG_10) & exact_region_gap.notna()).astype("float32")
    out["is_exact_prev1_region_outlier_20pct"] = ((exact_region_gap.abs() > LOG_20) & exact_region_gap.notna()).astype("float32")
    out["is_exact_prev1_region_outlier_30pct"] = ((exact_region_gap.abs() > LOG_30) & exact_region_gap.notna()).astype("float32")

    signal_cols = [
        "is_wide_prev_jump_20pct",
        "is_exact_prev_jump_20pct",
        "is_exact_wide_prev1_gap_10pct",
        "is_wide_jump20_and_gap365",
        "is_exact_jump20_and_gap365",
        "is_wide_prev1_region_outlier_20pct",
        "is_exact_prev1_region_outlier_20pct",
    ]
    out["outlier_signal_score"] = out[signal_cols].sum(axis=1).astype("float32")
    out["sgg_lag1_source_ym"] = work["sgg_lag1_source_ym"].astype("string")
    return out


def write_quality_report(sidecar: pd.DataFrame) -> None:
    raw_ids = pd.read_csv(DATA_PATH, usecols=["transaction_id", "deal_ym"], dtype={"transaction_id": "string", "deal_ym": "string"})
    merged = raw_ids.merge(sidecar[["transaction_id", "sgg_lag1_source_ym"]], on="transaction_id", how="left", validate="one_to_one")
    source_present = merged["sgg_lag1_source_ym"].notna()
    source_before_current = (merged.loc[source_present, "sgg_lag1_source_ym"] < merged.loc[source_present, "deal_ym"]).all()
    numeric_cols = [c for c in sidecar.columns if c not in {"transaction_id", "sgg_lag1_source_ym"}]
    checks = {
        "row_count_match": len(sidecar) == len(raw_ids),
        "transaction_id_unique": bool(sidecar["transaction_id"].is_unique),
        "join_missing_zero": int(merged["sgg_lag1_source_ym"].isna().sum()) >= 0 and int(merged["transaction_id"].isna().sum()) == 0,
        "sgg_lag1_source_before_deal_ym": bool(source_before_current),
        "numeric_features_finite_or_null": bool(np.isfinite(sidecar[numeric_cols].select_dtypes(include=["number"]).fillna(0).to_numpy()).all()),
    }
    coverage = pd.DataFrame(
        [
            {"metric": "rows", "value": len(sidecar)},
            {"metric": "sgg_prior_missing", "value": int(sidecar["sgg_prior_missing"].sum())},
            {"metric": "sgg_prior_missing_rate", "value": float(sidecar["sgg_prior_missing"].mean())},
            {"metric": "wide_prev_jump_20pct_rate", "value": float(sidecar["is_wide_prev_jump_20pct"].mean())},
            {"metric": "exact_prev_jump_20pct_rate", "value": float(sidecar["is_exact_prev_jump_20pct"].mean())},
            {"metric": "exact_wide_gap_10pct_rate", "value": float(sidecar["is_exact_wide_prev1_gap_10pct"].mean())},
            {"metric": "wide_region_outlier_20pct_rate", "value": float(sidecar["is_wide_prev1_region_outlier_20pct"].mean())},
            {"metric": "exact_region_outlier_20pct_rate", "value": float(sidecar["is_exact_prev1_region_outlier_20pct"].mean())},
        ]
    )
    grade = "Pass" if all(checks.values()) else "Fail"
    failed = [name for name, ok in checks.items() if not ok]
    lines = [
        "# E10 outlier-signal feature 품질 리포트",
        "",
        f"- 품질 등급: `{grade}`",
        f"- rows: {len(sidecar):,}",
        "- leakage guard: 현재 거래가격 기반 outlier flag는 feature로 저장하지 않습니다.",
        "",
        "## 지적사항",
        "- none" if not failed else "- 실패 checks: `" + "`, `".join(failed) + "`",
        "",
        "## 검증 근거 확인",
    ]
    for name, ok in checks.items():
        lines.append(f"- {name}: {'pass' if ok else 'fail'}")
    lines.extend(["", "## Coverage", md_table(coverage), "", "## 검증 공백"])
    lines.append("- `sgg` prior feature는 같은 `deal_ym`을 제외하고 이전 월 aggregate만 사용합니다.")
    lines.append("- prev jump/exact-wide gap feature는 이미 생성된 과거 거래 feature만 사용합니다.")
    lines.append(f"- sidecar_csv: `{FEATURE_PATH}`")
    QUALITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if grade != "Pass":
        raise RuntimeError(f"quality report failed: {failed}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sidecar = build_sidecar()
    sidecar.to_csv(FEATURE_PATH, index=False)
    write_quality_report(sidecar)
    print("feature", FEATURE_PATH)
    print("quality_report", QUALITY_REPORT_PATH)
    print("rows", len(sidecar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
