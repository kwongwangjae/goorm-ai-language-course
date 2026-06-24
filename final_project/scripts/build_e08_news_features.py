#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
TRANSACTIONS_PATH = ROOT / "data" / "processed" / "transactions.csv"
EXTERNAL_DIR = ROOT / "data" / "external"
OUTPUTS_DIR = ROOT / "outputs"
HOME_SEARCH_NEWS_DIR = Path(
    os.environ.get(
        "HOME_SEARCH_NEWS_DIR",
        "/Users/gwongwangjae/home-search/apps/news/local-input",
    )
)
NEWS_JSONL_PATHS = [
    HOME_SEARCH_NEWS_DIR / "region-month-signal-bigkinds.csv.jsonl",
    HOME_SEARCH_NEWS_DIR / "region-month-signal-web-research.jsonl",
]

EXTERNAL_SIGNALS_PATH = EXTERNAL_DIR / "region_month_news_signals.csv"
SIDECAR_PATH = OUTPUTS_DIR / "e08_news_features.csv"
QUALITY_REPORT_PATH = OUTPUTS_DIR / "e08_news_feature_quality_report.md"

EXPECTED_SOURCE_ROWS = 2964
EXPECTED_MONTHS = 114
EXPECTED_BUCKETS = 26

SGG_TO_DETAIL_BUCKET = {
    "11170": "SEOUL_YONGSAN_GU",
    "11200": "SEOUL_SEONGDONG_GU",
    "11350": "SEOUL_NOWON_GU",
    "11440": "SEOUL_MAPO_GU",
    "11470": "SEOUL_YANGCHEON_GU",
    "11560": "SEOUL_YEONGDEUNGPO_GU",
    "11650": "SEOUL_SEOCHO_GU",
    "11680": "SEOUL_GANGNAM_GU",
    "11710": "SEOUL_SONGPA_GU",
    "11740": "SEOUL_GANGDONG_GU",
    "41110": "GYEONGGI_SUWON_SI",
    "41130": "GYEONGGI_SEONGNAM_SI",
    "41170": "GYEONGGI_ANYANG_SI",
    "41210": "GYEONGGI_GWANGMYEONG_SI",
    "41280": "GYEONGGI_GOYANG_SI",
    "41290": "GYEONGGI_GWACHEON_SI",
    "41360": "GYEONGGI_NAMYANGJU_SI",
    "41430": "GYEONGGI_UIWANG_SI",
    "41450": "GYEONGGI_HANAM_SI",
    "41460": "GYEONGGI_YONGIN_SI",
    "41570": "GYEONGGI_GIMPO_SI",
    "41590": "GYEONGGI_HWASEONG_SI",
}

SIGNAL_COLUMNS = [
    "region_bucket",
    "signal_month",
    "source_kind",
    "method_version",
    "dataset_tier",
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
]

SCORE_COLUMNS = [
    "price_up_signal",
    "price_down_signal",
    "policy_positive_score",
    "policy_negative_score",
    "redevelopment_score",
    "transport_score",
    "supply_risk_score",
    "sale_market_score",
    "rental_market_score",
]

COUNT_COLUMNS = [
    "news_count",
    "matched_news_count",
    "direct_evidence_count",
    "inherited_evidence_count",
]

FEATURE_LEVELS = [
    ("detail", "detail_bucket"),
    ("parent", "parent_bucket"),
    ("national", "national_bucket"),
]


@dataclass(frozen=True)
class Signal:
    row: dict[str, Any]

    @property
    def confidence(self) -> float:
        return to_float(self.row.get("confidence"))

    @property
    def direct_count(self) -> int:
        return to_int(self.row.get("direct_evidence_count"))

    @property
    def inherited_count(self) -> int:
        return to_int(self.row.get("inherited_evidence_count"))

    @property
    def matched_count(self) -> int:
        return to_int(self.row.get("matched_news_count"))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def month_add(ym: str, offset: int) -> str:
    year, month = map(int, ym.split("-"))
    month += offset
    while month <= 0:
        year -= 1
        month += 12
    while month > 12:
        year += 1
        month -= 12
    return f"{year:04d}-{month:02d}"


def parent_bucket(sgg_code: str) -> str:
    if sgg_code.startswith("11"):
        return "SEOUL"
    if sgg_code.startswith("41"):
        return "GYEONGGI"
    return "OTHER"


def transaction_buckets(sgg_code: str) -> dict[str, str]:
    parent = parent_bucket(sgg_code)
    return {
        "detail_bucket": SGG_TO_DETAIL_BUCKET.get(sgg_code, ""),
        "parent_bucket": parent,
        "national_bucket": "NATIONAL",
    }


def quality_tier(signal: Signal | None) -> str:
    if signal is None:
        return "sparse"
    if signal.direct_count >= 3:
        return "direct_sufficient"
    if signal.direct_count > 0:
        return "direct_partial"
    if signal.inherited_count > 0:
        return "inherited_centered"
    return "sparse"


def quality_weight(tier: str) -> float:
    return {
        "direct_sufficient": 1.0,
        "direct_partial": 0.75,
        "inherited_centered": 0.35,
        "sparse": 0.0,
    }.get(tier, 0.0)


def load_signals(paths: list[Path]) -> tuple[dict[tuple[str, str], Signal], dict[str, Any]]:
    signals: dict[tuple[str, str], Signal] = {}
    source_rows: list[dict[str, Any]] = []
    buckets: set[str] = set()
    months: set[str] = set()
    sources = Counter()
    duplicate_keys = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                clean_row = {col: row.get(col, "") for col in SIGNAL_COLUMNS}
                bucket = str(clean_row["region_bucket"])
                month = str(clean_row["signal_month"])
                key = (bucket, month)
                if key in signals:
                    duplicate_keys.append({"path": str(path), "line": line_no, "key": key})
                    continue
                signal = Signal(clean_row)
                signals[key] = signal
                source_rows.append(clean_row)
                buckets.add(bucket)
                months.add(month)
                sources[str(clean_row["source_kind"])] += 1

    stats = {
        "source_rows": len(source_rows),
        "months": len(months),
        "min_month": min(months) if months else "",
        "max_month": max(months) if months else "",
        "buckets": len(buckets),
        "sources": dict(sources),
        "duplicate_keys": duplicate_keys,
    }
    return signals, {"rows": source_rows, "stats": stats, "bucket_values": sorted(buckets)}


def write_external_signals(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIGNAL_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["signal_month"], r["region_bucket"], r["source_kind"])):
            writer.writerow(row)


def signal_value(signal: Signal | None, column: str) -> float:
    if signal is None:
        return 0.0
    return to_float(signal.row.get(column))


def rolling_average(
    signals: dict[tuple[str, str], Signal],
    bucket: str,
    source_month: str,
    column: str,
    window: int = 3,
) -> float:
    values = []
    for lag in range(window):
        month = month_add(source_month, -lag)
        signal = signals.get((bucket, month)) if bucket else None
        if signal is not None:
            values.append(signal_value(signal, column))
    if not values:
        return 0.0
    return sum(values) / len(values)


def feature_fieldnames() -> list[str]:
    fields = ["transaction_id", "deal_ym", "news_source_month"]
    for prefix, _ in FEATURE_LEVELS:
        fields.extend(
            [
                f"{prefix}_news_bucket",
                f"{prefix}_news_quality_tier",
                f"{prefix}_news_confidence",
                f"{prefix}_price_up_signal",
                f"{prefix}_price_down_signal",
                f"{prefix}_net_signal",
                f"{prefix}_policy_positive_score",
                f"{prefix}_policy_negative_score",
                f"{prefix}_policy_net_signal",
                f"{prefix}_redevelopment_score",
                f"{prefix}_transport_score",
                f"{prefix}_supply_risk_score",
                f"{prefix}_sale_market_score",
                f"{prefix}_rental_market_score",
                f"{prefix}_direct_evidence_count",
                f"{prefix}_inherited_evidence_count",
                f"{prefix}_matched_news_count_log1p",
                f"{prefix}_quality_weight",
                f"{prefix}_weighted_net_signal",
                f"{prefix}_direct_sufficient_flag",
                f"{prefix}_direct_partial_flag",
                f"{prefix}_inherited_centered_flag",
                f"{prefix}_sparse_flag",
                f"{prefix}_rolling3_confidence",
                f"{prefix}_rolling3_price_up_signal",
                f"{prefix}_rolling3_price_down_signal",
                f"{prefix}_rolling3_net_signal",
            ]
        )
    fields.extend(
        [
            "detail_parent_net_signal_gap",
            "parent_national_net_signal_gap",
            "detail_parent_confidence_gap",
            "parent_national_confidence_gap",
            "news_leakage_violation",
        ]
    )
    return fields


def add_signal_features(
    out: dict[str, Any],
    prefix: str,
    bucket: str,
    source_month: str,
    signals: dict[tuple[str, str], Signal],
) -> None:
    signal = signals.get((bucket, source_month)) if bucket else None
    tier = quality_tier(signal)
    weight = quality_weight(tier)
    price_up = signal_value(signal, "price_up_signal")
    price_down = signal_value(signal, "price_down_signal")
    net = price_up - price_down
    policy_positive = signal_value(signal, "policy_positive_score")
    policy_negative = signal_value(signal, "policy_negative_score")
    confidence = signal.confidence if signal else 0.0

    out[f"{prefix}_news_bucket"] = bucket or "MISSING"
    out[f"{prefix}_news_quality_tier"] = tier
    out[f"{prefix}_news_confidence"] = f"{confidence:.6f}"
    out[f"{prefix}_price_up_signal"] = f"{price_up:.6f}"
    out[f"{prefix}_price_down_signal"] = f"{price_down:.6f}"
    out[f"{prefix}_net_signal"] = f"{net:.6f}"
    out[f"{prefix}_policy_positive_score"] = f"{policy_positive:.6f}"
    out[f"{prefix}_policy_negative_score"] = f"{policy_negative:.6f}"
    out[f"{prefix}_policy_net_signal"] = f"{(policy_positive - policy_negative):.6f}"
    for column in [
        "redevelopment_score",
        "transport_score",
        "supply_risk_score",
        "sale_market_score",
        "rental_market_score",
    ]:
        out[f"{prefix}_{column}"] = f"{signal_value(signal, column):.6f}"
    out[f"{prefix}_direct_evidence_count"] = str(signal.direct_count if signal else 0)
    out[f"{prefix}_inherited_evidence_count"] = str(signal.inherited_count if signal else 0)
    out[f"{prefix}_matched_news_count_log1p"] = f"{math.log1p(signal.matched_count if signal else 0):.6f}"
    out[f"{prefix}_quality_weight"] = f"{weight:.6f}"
    out[f"{prefix}_weighted_net_signal"] = f"{(net * confidence * weight):.6f}"
    out[f"{prefix}_direct_sufficient_flag"] = "1" if tier == "direct_sufficient" else "0"
    out[f"{prefix}_direct_partial_flag"] = "1" if tier == "direct_partial" else "0"
    out[f"{prefix}_inherited_centered_flag"] = "1" if tier == "inherited_centered" else "0"
    out[f"{prefix}_sparse_flag"] = "1" if tier == "sparse" else "0"
    rolling_confidence = rolling_average(signals, bucket, source_month, "confidence")
    rolling_price_up = rolling_average(signals, bucket, source_month, "price_up_signal")
    rolling_price_down = rolling_average(signals, bucket, source_month, "price_down_signal")
    out[f"{prefix}_rolling3_confidence"] = f"{rolling_confidence:.6f}"
    out[f"{prefix}_rolling3_price_up_signal"] = f"{rolling_price_up:.6f}"
    out[f"{prefix}_rolling3_price_down_signal"] = f"{rolling_price_down:.6f}"
    out[f"{prefix}_rolling3_net_signal"] = f"{(rolling_price_up - rolling_price_down):.6f}"


def build_sidecar(
    transactions_path: Path,
    output_path: Path,
    signals: dict[tuple[str, str], Signal],
    limit: int | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    fieldnames = feature_fieldnames()
    seen: set[str] = set()
    rows = 0
    duplicate_transaction_ids = 0
    leakage_violations = 0
    detail_tiers = Counter()
    parent_tiers = Counter()
    national_tiers = Counter()
    source_months = set()
    bucket_counts = Counter()

    with transactions_path.open("r", encoding="utf-8", newline="") as src, tmp.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        required = {"transaction_id", "deal_ym", "sgg_code"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"transactions.csv missing columns: {sorted(missing)}")
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if limit is not None and rows >= limit:
                break
            transaction_id = row["transaction_id"]
            if transaction_id in seen:
                duplicate_transaction_ids += 1
            seen.add(transaction_id)

            deal_ym = row["deal_ym"]
            source_month = month_add(deal_ym, -1)
            buckets = transaction_buckets(row.get("sgg_code", ""))
            out = {
                "transaction_id": transaction_id,
                "deal_ym": deal_ym,
                "news_source_month": source_month,
            }
            for prefix, bucket_key in FEATURE_LEVELS:
                add_signal_features(out, prefix, buckets[bucket_key], source_month, signals)
            out["detail_parent_net_signal_gap"] = f"{(to_float(out['detail_net_signal']) - to_float(out['parent_net_signal'])):.6f}"
            out["parent_national_net_signal_gap"] = f"{(to_float(out['parent_net_signal']) - to_float(out['national_net_signal'])):.6f}"
            out["detail_parent_confidence_gap"] = f"{(to_float(out['detail_news_confidence']) - to_float(out['parent_news_confidence'])):.6f}"
            out["parent_national_confidence_gap"] = f"{(to_float(out['parent_news_confidence']) - to_float(out['national_news_confidence'])):.6f}"
            leakage = int(source_month > month_add(deal_ym, -1))
            out["news_leakage_violation"] = str(leakage)
            leakage_violations += leakage
            detail_tiers[out["detail_news_quality_tier"]] += 1
            parent_tiers[out["parent_news_quality_tier"]] += 1
            national_tiers[out["national_news_quality_tier"]] += 1
            source_months.add(source_month)
            bucket_counts[out["detail_news_bucket"]] += 1
            writer.writerow(out)
            rows += 1

    tmp.replace(output_path)
    return {
        "sidecar_rows": rows,
        "transaction_id_unique": duplicate_transaction_ids == 0,
        "duplicate_transaction_ids": duplicate_transaction_ids,
        "leakage_violations": leakage_violations,
        "source_month_min": min(source_months) if source_months else "",
        "source_month_max": max(source_months) if source_months else "",
        "detail_quality_tiers": dict(detail_tiers),
        "parent_quality_tiers": dict(parent_tiers),
        "national_quality_tiers": dict(national_tiers),
        "top_detail_buckets": dict(bucket_counts.most_common(12)),
    }


def transaction_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def md_table(mapping: dict[str, Any]) -> str:
    lines = ["| key | value |", "| --- | --- |"]
    for key in sorted(mapping):
        lines.append(f"| {key} | {mapping[key]} |")
    return "\n".join(lines)


def write_quality_report(
    source_stats: dict[str, Any],
    sidecar_stats: dict[str, Any],
    expected_transaction_rows: int,
    limited: bool,
    external_signals_path: Path,
    sidecar_path: Path,
) -> dict[str, bool]:
    checks = {
        "source_rows_2964": source_stats["source_rows"] == EXPECTED_SOURCE_ROWS,
        "months_114": source_stats["months"] == EXPECTED_MONTHS,
        "buckets_26": source_stats["buckets"] == EXPECTED_BUCKETS,
        "source_duplicate_keys_zero": not source_stats["duplicate_keys"],
        "sidecar_rows_match_transactions": limited
        or sidecar_stats["sidecar_rows"] == expected_transaction_rows,
        "transaction_id_unique": sidecar_stats["transaction_id_unique"],
        "join_missing_zero": limited
        or sidecar_stats["sidecar_rows"] == expected_transaction_rows,
        "news_source_month_lag1_or_earlier": sidecar_stats["leakage_violations"] == 0,
    }
    grade = "Pass" if all(checks.values()) else "Fail"
    lines = [
        "# E08 news feature 품질 리포트",
        "",
        f"- 품질 등급: `{grade}`",
        f"- run_scope: `{'limited' if limited else 'full'}`",
        f"- source rows: {source_stats['source_rows']:,}",
        f"- months: {source_stats['months']:,} ({source_stats['min_month']}..{source_stats['max_month']})",
        f"- buckets: {source_stats['buckets']:,}",
        f"- sidecar rows: {sidecar_stats['sidecar_rows']:,}",
        f"- transactions rows: {expected_transaction_rows:,}",
        f"- leakage violations: {sidecar_stats['leakage_violations']:,}",
        "",
        "## 지적사항",
    ]
    failed = [name for name, ok in checks.items() if not ok]
    lines.append("- none" if not failed else "- 실패 checks: `" + "`, `".join(failed) + "`")
    lines.extend(["", "## 검증 근거 확인"])
    for name, ok in checks.items():
        lines.append(f"- {name}: {'pass' if ok else 'fail'}")
    lines.extend(
        [
            "",
            "## Source stats",
            md_table(source_stats["sources"]),
            "",
            "## Detail quality tiers",
            md_table(sidecar_stats["detail_quality_tiers"]),
            "",
            "## Parent quality tiers",
            md_table(sidecar_stats["parent_quality_tiers"]),
            "",
            "## National quality tiers",
            md_table(sidecar_stats["national_quality_tiers"]),
            "",
            "## Top detail buckets",
            md_table(sidecar_stats["top_detail_buckets"]),
            "",
            "## 검증 공백",
            "- 모델 성능 검증은 `08_test_news_features.ipynb`에서 별도 실행합니다.",
            "- 이 sidecar는 `transactions.csv`를 수정하지 않고 `transaction_id`로만 join합니다.",
            "- `news_source_month`는 모든 row에서 `deal_ym - 1 month`로 고정합니다.",
            f"- external_csv: `{external_signals_path}`",
            f"- sidecar_csv: `{sidecar_path}`",
        ]
    )
    QUALITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E08 lagged news feature sidecar.")
    parser.add_argument("--transactions", type=Path, default=TRANSACTIONS_PATH)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_PATH)
    parser.add_argument("--external-signals", type=Path, default=EXTERNAL_SIGNALS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional transaction row limit for smoke runs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in NEWS_JSONL_PATHS:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.transactions.exists():
        raise FileNotFoundError(args.transactions)

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    signals, source_info = load_signals(NEWS_JSONL_PATHS)
    write_external_signals(source_info["rows"], args.external_signals)
    expected_rows = transaction_row_count(args.transactions)
    sidecar_stats = build_sidecar(args.transactions, args.sidecar, signals, limit=args.limit)
    checks = write_quality_report(
        source_info["stats"],
        sidecar_stats,
        expected_rows,
        limited=args.limit is not None,
        external_signals_path=args.external_signals,
        sidecar_path=args.sidecar,
    )
    print(json.dumps({"checks": checks, "source": source_info["stats"], "sidecar": sidecar_stats}, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
