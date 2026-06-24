#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


current_dir = Path.cwd()
if current_dir.name == "final_project":
    PROJECT_DIR = current_dir
elif (current_dir / "final_project").exists():
    PROJECT_DIR = current_dir / "final_project"
else:
    PROJECT_DIR = Path("/Users/gwongwangjae/goorm-ai-language-course/final_project")

SCRIPT_PATH = PROJECT_DIR / "scripts" / "build_transactions_dataset.py"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "transactions.csv"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FEATURE_PATH = OUTPUT_DIR / "e09_exact_prev_features.csv"
QUALITY_REPORT_PATH = OUTPUT_DIR / "e09_exact_prev_feature_quality_report.md"

EXACT_AREA_TOLERANCE_M2 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

RAW_COLUMNS = [
    "transaction_id",
    "exact_prev1_price_per_m2",
    "exact_prev1_missing",
    "exact_prev1_gap_days",
    "exact_prev1_source_deal_date",
    "exact_prev1_source_area_m2",
    "exact_prev2_price_per_m2",
    "exact_prev2_missing",
    "exact_prev2_gap_days",
    "exact_prev2_source_deal_date",
    "exact_prev2_source_area_m2",
]

SIDECAR_COLUMNS = [
    *RAW_COLUMNS,
    "log_exact_prev1_price_per_m2",
    "log_exact_prev2_price_per_m2",
    "exact_prev1_prev2_log_return",
    "exact_prev1_prev2_gap_days",
    "exact_prev1_area_abs_diff",
    "exact_prev2_area_abs_diff",
    "wide_prev1_present_exact_missing",
    "exact_prev1_present_wide_missing",
]


@dataclass(frozen=True)
class PrevHist:
    area: float
    deal_date: date
    ppm: float
    seq: int


def load_builder():
    spec = importlib.util.spec_from_file_location("transactions_dataset_builder", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load builder: {SCRIPT_PATH}")
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    return builder


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def add_hist(hist, row: dict[str, Any], seq: int) -> int:
    if row["is_cancelled"] or not row["complex_id"] or not row["deal_date_obj"]:
        return seq
    area = to_float(row["area_m2"])
    ppm = to_float(row["price_per_m2"])
    if area is None or area <= 0 or ppm is None or ppm <= 0:
        return seq
    seq += 1
    hist[row["complex_id"]][round(area, 1)].append(PrevHist(area, row["deal_date_obj"], ppm, seq))
    return seq


def find_exact_prev_two(hist, complex_id: str, area_value: Any, deal_dt: date | None) -> list[PrevHist]:
    area = to_float(area_value)
    if area is None or area <= 0 or not complex_id or not deal_dt:
        return []
    buckets = hist.get(complex_id)
    if not buckets:
        return []
    candidates: list[PrevHist] = []
    center = round(area, 1)
    for bucket10 in range(int(math.floor((area - EXACT_AREA_TOLERANCE_M2) * 10)), int(math.ceil((area + EXACT_AREA_TOLERANCE_M2) * 10)) + 1):
        bucket = bucket10 / 10.0
        items = buckets.get(bucket)
        if not items:
            continue
        found = 0
        for item in reversed(items):
            if item.deal_date >= deal_dt:
                continue
            if abs(item.area - area) <= EXACT_AREA_TOLERANCE_M2:
                candidates.append(item)
                found += 1
                if found >= 2:
                    break
    # The center variable exists to make the rounded-bucket intent explicit.
    _ = center
    candidates.sort(key=lambda item: (item.deal_date, item.seq), reverse=True)
    return candidates[:2]


def raw_row(transaction_id: str, deal_dt: date, current_area: Any, prevs: list[PrevHist]) -> dict[str, str]:
    area = to_float(current_area)
    prev1 = prevs[0] if len(prevs) >= 1 else None
    prev2 = prevs[1] if len(prevs) >= 2 else None

    def values(prefix: str, item: PrevHist | None) -> dict[str, str]:
        if item is None:
            return {
                f"{prefix}_price_per_m2": "",
                f"{prefix}_missing": "1",
                f"{prefix}_gap_days": "",
                f"{prefix}_source_deal_date": "",
                f"{prefix}_source_area_m2": "",
            }
        return {
            f"{prefix}_price_per_m2": f"{item.ppm:.12f}",
            f"{prefix}_missing": "0",
            f"{prefix}_gap_days": str((deal_dt - item.deal_date).days),
            f"{prefix}_source_deal_date": item.deal_date.isoformat(),
            f"{prefix}_source_area_m2": f"{item.area:.6f}",
        }

    out = {"transaction_id": transaction_id}
    out.update(values("exact_prev1", prev1))
    out.update(values("exact_prev2", prev2))
    return out


def write_raw_sidecar(raw_tmp_path: Path) -> dict[str, Any]:
    builder = load_builder()
    paths = sorted(INTERIM_DIR.glob("transactions_base_*.csv.gz"))
    if not paths:
        raise RuntimeError(f"no interim chunks found: {INTERIM_DIR}")
    hist = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    current_day = None
    day_records: list[dict[str, Any]] = []
    seq = 0
    stats: dict[str, Any] = {
        "base_rows": 0,
        "sidecar_rows": 0,
        "exact_prev1_missing_rows": 0,
        "exact_prev2_missing_rows": 0,
        "collision_rows": 0,
        "source_date_failures": 0,
    }

    def flush(records: list[dict[str, Any]], writer: csv.DictWriter) -> None:
        nonlocal seq
        if not records:
            return
        for row in records:
            prevs = find_exact_prev_two(hist, row["complex_id"], row["area_m2"], row["deal_date_obj"])
            if not row["deal_date_obj"] or row["deal_date_obj"] < date(2019, 1, 1):
                continue
            reason = builder.exclusion(row)
            if reason:
                continue
            tid = row["transaction_id"]
            if tid in seen:
                tid = f"{tid}_{row['trade_id']}"
                row["transaction_id"] = tid
                stats["collision_rows"] += 1
            if tid in seen:
                raise RuntimeError(f"duplicate transaction_id after collision handling: {tid}")
            seen.add(tid)
            if len(prevs) < 1:
                stats["exact_prev1_missing_rows"] += 1
            if len(prevs) < 2:
                stats["exact_prev2_missing_rows"] += 1
            if any((row["deal_date_obj"] - prev.deal_date).days <= 0 for prev in prevs):
                stats["source_date_failures"] += 1
            writer.writerow(raw_row(tid, row["deal_date_obj"], row["area_m2"], prevs))
            stats["sidecar_rows"] += 1
        for row in records:
            seq = add_hist(hist, row, seq)

    with raw_tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for raw in builder.iter_rows(paths):
            stats["base_rows"] += 1
            row = builder.prepare(raw)
            deal_dt = row["deal_date_obj"]
            if current_day is None:
                current_day = deal_dt
            if deal_dt != current_day:
                flush(day_records, writer)
                day_records = []
                current_day = deal_dt
            day_records.append(row)
        flush(day_records, writer)
    return stats


def md_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    x = frame.copy()
    for col in x.select_dtypes(include=["float", "float32", "float64"]).columns:
        x[col] = x[col].map(lambda v: format(v, floatfmt) if pd.notna(v) else "")
    x = x.astype("string").fillna("")
    lines = ["| " + " | ".join(x.columns) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in x.values.tolist()]
    return "\n".join(lines)


def build_sidecar(raw_tmp_path: Path, stats: dict[str, Any]) -> None:
    transactions = pd.read_csv(
        DATA_PATH,
        usecols=["transaction_id", "deal_date", "area_m2", "complex_prev_price_per_m2", "complex_prev_missing", "prev_deal_gap_days"],
        dtype={
            "transaction_id": "string",
            "area_m2": "float32",
            "complex_prev_price_per_m2": "float32",
            "complex_prev_missing": "Int8",
            "prev_deal_gap_days": "float32",
        },
        parse_dates=["deal_date"],
    )
    raw = pd.read_csv(
        raw_tmp_path,
        dtype={"transaction_id": "string", "exact_prev1_missing": "Int8", "exact_prev2_missing": "Int8"},
        parse_dates=["exact_prev1_source_deal_date", "exact_prev2_source_deal_date"],
    )
    if not raw["transaction_id"].is_unique:
        raise RuntimeError("raw exact prev sidecar transaction_id is not unique")
    merged = transactions.merge(raw, on="transaction_id", how="left", validate="one_to_one")
    join_missing = int(merged["exact_prev1_missing"].isna().sum())
    if join_missing:
        raise RuntimeError(f"join missing rows: {join_missing}")

    for col in ["exact_prev1_price_per_m2", "exact_prev2_price_per_m2", "exact_prev1_gap_days", "exact_prev2_gap_days", "exact_prev1_source_area_m2", "exact_prev2_source_area_m2"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    prev1_price = merged["exact_prev1_price_per_m2"]
    prev2_price = merged["exact_prev2_price_per_m2"]
    merged["log_exact_prev1_price_per_m2"] = prev1_price.where(prev1_price > 0).map(lambda v: math.log(v) if pd.notna(v) else float("nan"))
    merged["log_exact_prev2_price_per_m2"] = prev2_price.where(prev2_price > 0).map(lambda v: math.log(v) if pd.notna(v) else float("nan"))
    merged["exact_prev1_prev2_log_return"] = merged["log_exact_prev1_price_per_m2"] - merged["log_exact_prev2_price_per_m2"]
    merged["exact_prev1_prev2_gap_days"] = merged["exact_prev2_gap_days"] - merged["exact_prev1_gap_days"]
    merged["exact_prev1_area_abs_diff"] = (merged["area_m2"].astype("float64") - merged["exact_prev1_source_area_m2"]).abs()
    merged["exact_prev2_area_abs_diff"] = (merged["area_m2"].astype("float64") - merged["exact_prev2_source_area_m2"]).abs()
    wide_present = merged["complex_prev_missing"].astype("Int8").eq(0)
    exact_present = merged["exact_prev1_missing"].astype("Int8").eq(0)
    merged["wide_prev1_present_exact_missing"] = (wide_present & ~exact_present).astype("Int8")
    merged["exact_prev1_present_wide_missing"] = (exact_present & ~wide_present).astype("Int8")

    sidecar = merged[SIDECAR_COLUMNS].copy()
    sidecar.to_csv(FEATURE_PATH, index=False)

    exact1_present = sidecar["exact_prev1_missing"].astype("Int8").eq(0)
    exact2_present = sidecar["exact_prev2_missing"].astype("Int8").eq(0)
    source1_dates = pd.to_datetime(sidecar["exact_prev1_source_deal_date"], errors="coerce")
    source2_dates = pd.to_datetime(sidecar["exact_prev2_source_deal_date"], errors="coerce")
    deal_dates = pd.to_datetime(merged["deal_date"], errors="coerce")
    checks = {
        "row_count_match": len(sidecar) == len(transactions),
        "transaction_id_unique": sidecar["transaction_id"].is_unique,
        "join_missing_zero": join_missing == 0,
        "exact_prev1_present_fields": bool((sidecar.loc[exact1_present, "exact_prev1_price_per_m2"].gt(0) & sidecar.loc[exact1_present, "exact_prev1_gap_days"].gt(0) & source1_dates[exact1_present].notna()).all()),
        "exact_prev2_present_fields": bool((sidecar.loc[exact2_present, "exact_prev2_price_per_m2"].gt(0) & sidecar.loc[exact2_present, "exact_prev2_gap_days"].gt(0) & source2_dates[exact2_present].notna()).all()),
        "exact_prev1_source_before_deal": bool((source1_dates[exact1_present] < deal_dates[exact1_present]).all()),
        "exact_prev2_source_before_deal": bool((source2_dates[exact2_present] < deal_dates[exact2_present]).all()),
        "exact_prev1_area_within_tolerance": bool(sidecar.loc[exact1_present, "exact_prev1_area_abs_diff"].le(EXACT_AREA_TOLERANCE_M2 + 1e-4).all()),
        "exact_prev2_area_within_tolerance": bool(sidecar.loc[exact2_present, "exact_prev2_area_abs_diff"].le(EXACT_AREA_TOLERANCE_M2 + 1e-4).all()),
        "exact_prev1_prev2_gap_non_negative": bool(sidecar.loc[exact2_present, "exact_prev1_prev2_gap_days"].ge(0).all()),
        "source_date_failures_zero": int(stats.get("source_date_failures", 0)) == 0,
    }
    grade = "Pass" if all(checks.values()) else "Fail"

    coverage = pd.DataFrame(
        [
            {"metric": "rows", "value": len(sidecar)},
            {"metric": "exact_prev1_present", "value": int(exact1_present.sum())},
            {"metric": "exact_prev1_missing", "value": int((~exact1_present).sum())},
            {"metric": "exact_prev1_missing_rate", "value": float((~exact1_present).mean())},
            {"metric": "exact_prev2_present", "value": int(exact2_present.sum())},
            {"metric": "exact_prev2_missing", "value": int((~exact2_present).sum())},
            {"metric": "exact_prev2_missing_rate", "value": float((~exact2_present).mean())},
            {"metric": "wide_prev1_present_exact_missing", "value": int(sidecar["wide_prev1_present_exact_missing"].sum())},
            {"metric": "exact_prev1_present_wide_missing", "value": int(sidecar["exact_prev1_present_wide_missing"].sum())},
        ]
    )
    yearly = sidecar.assign(year=deal_dates.dt.year, exact_prev1_present=exact1_present, exact_prev2_present=exact2_present).groupby("year", dropna=False).agg(
        rows=("transaction_id", "size"),
        exact_prev1_present=("exact_prev1_present", "sum"),
        exact_prev2_present=("exact_prev2_present", "sum"),
    ).reset_index()
    yearly["exact_prev1_missing_rate"] = 1.0 - yearly["exact_prev1_present"] / yearly["rows"]
    yearly["exact_prev2_missing_rate"] = 1.0 - yearly["exact_prev2_present"] / yearly["rows"]

    base_rows = stats.get("base_rows")
    collision_rows = stats.get("collision_rows")
    source_rows_label = f"{base_rows:,}" if base_rows else "not retained"
    collision_rows_label = f"{collision_rows:,}" if collision_rows else "not retained"

    lines = [
        "# E09 exact-area prev feature 품질 리포트",
        "",
        f"- 품질 등급: `{grade}`",
        f"- exact_area_tolerance_m2: {EXACT_AREA_TOLERANCE_M2}",
        f"- rows: {len(sidecar):,}",
        f"- source rows scanned: {source_rows_label}",
        f"- collision rows: {collision_rows_label}",
        "",
        "## 지적사항",
    ]
    failed = [name for name, ok in checks.items() if not ok]
    lines.append("- none" if not failed else "- 실패 checks: `" + "`, `".join(failed) + "`")
    lines.extend(["", "## 검증 근거 확인"])
    for name, ok in checks.items():
        lines.append(f"- {name}: {'pass' if ok else 'fail'}")
    lines.extend(["", "## Coverage", md_table(coverage), "", "## Yearly coverage", md_table(yearly), "", "## 검증 공백"])
    lines.append("- exact-area prev는 `transactions.csv`를 수정하지 않는 sidecar feature입니다.")
    lines.append("- source 거래는 항상 현재 거래일보다 이전 거래만 허용합니다.")
    lines.append("- 동일 거래일은 day buffer flush 후 history에 추가하므로 현재 거래의 history로 쓰이지 않습니다.")
    lines.append(f"- sidecar_csv: `{FEATURE_PATH}`")
    QUALITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if grade != "Pass":
        raise RuntimeError(f"quality report failed: {failed}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_tmp = OUTPUT_DIR / "e09_exact_prev_features.raw.tmp.csv"
    start = time.perf_counter()
    stats = write_raw_sidecar(raw_tmp)
    build_sidecar(raw_tmp, stats)
    raw_tmp.unlink(missing_ok=True)
    print("e09 exact prev sidecar built", stats, "seconds", round(time.perf_counter() - start, 2))
    print(FEATURE_PATH)
    print(QUALITY_REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
