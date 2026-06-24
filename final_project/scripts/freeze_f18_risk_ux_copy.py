#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
POLICY_DIR = REPO_DIR / "outputs" / "f18_confidence_interval_policy"
POLICY_JSON = POLICY_DIR / "confidence_policy.json"
OUTPUT_MD = POLICY_DIR / "RISK_CASE_UX_COPY.md"
OUTPUT_JSON = POLICY_DIR / "risk_case_copy.json"
OUTPUT_MANIFEST = POLICY_DIR / "risk_case_ux_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_copy(policy: dict[str, object]) -> dict[str, object]:
    return {
        "artifact": "f18_risk_case_ux_copy",
        "source_policy": policy["policy_name"],
        "price_model": policy["price_model"],
        "api_contract_impact": "none",
        "affected_ui_units": ["detail_drawer", "mobile_detail_sheet", "selected_complex_summary"],
        "do_not_show_on": ["region_marker", "default_complex_marker"],
        "risk_states": {
            "normal": {
                "condition": "confidence_tier in [high, medium] and resid_risk_tier not in [high, unknown]",
                "label": "예측 신뢰도 보통",
                "tone": "neutral",
                "primary": "최근 비교 이력을 기준으로 예측 범위를 산정했습니다.",
                "secondary": "실제 거래 조건에 따라 차이가 날 수 있습니다.",
                "interval": "show p90",
            },
            "low_history": {
                "condition": "confidence_tier == low and resid_risk_tier not in [high, unknown]",
                "label": "비교 이력 부족",
                "tone": "caution",
                "primary": "이 단지는 비교 이력이 적어 예측 범위를 넓게 봐야 합니다.",
                "secondary": "최근 실거래와 층, 동, 수리 상태를 함께 확인하세요.",
                "interval": "show p90 and make source visible",
            },
            "high_risk": {
                "condition": "resid_risk_tier == high",
                "label": "큰 오차 주의",
                "tone": "warning",
                "primary": "과거 오차 패턴상 실제 가격과 크게 차이날 가능성이 평균보다 높습니다.",
                "secondary": "가격은 참고값으로 보고 최근 실거래를 우선 확인하세요.",
                "interval": "show p95 by default, with p90 available as normal range",
            },
            "unknown": {
                "condition": "resid_risk_tier == unknown or no matching confidence report row before global fallback",
                "label": "신뢰도 확인 필요",
                "tone": "caution",
                "primary": "이 단지의 오차 이력이 부족해 보수적으로 해석해야 합니다.",
                "secondary": "지역 기준 범위로 대체 표시합니다.",
                "interval": "show fallback p95",
            },
        },
        "component_copy": {
            "section_title": "예측 신뢰도",
            "range_label": "예상 범위",
            "source_label": "기준",
            "source_templates": {
                "complex": "단지 기준",
                "legal_dong": "동 기준",
                "sgg": "구 기준",
                "sido": "시도 기준",
                "global": "전체 기준",
            },
            "range_helper": "범위는 과거 예측 오차의 p90/p95를 현재 예측가에 적용한 값입니다.",
            "disclaimer": "실거래 신고 지연, 층/동/향, 수리 상태, 특수거래는 별도 확인이 필요합니다.",
        },
        "layout_rules": {
            "desktop": [
                "Place the confidence row directly below the predicted price in the detail drawer.",
                "Use one compact status chip, one range row, and one short reason line.",
                "Keep trade history visible without adding a large warning card.",
            ],
            "mobile": [
                "Use a single-line summary above the fold in the bottom sheet.",
                "Put the longer helper text behind an expandable details row.",
            ],
            "marker": [
                "Do not add warning labels to default map markers.",
                "For selected markers only, a subtle warning border is acceptable when high_risk.",
            ],
        },
        "accessibility": {
            "meaning_not_color_only": True,
            "live_region": "polite when risk state changes after drawer data loads",
            "minimum_hit_area_px": 40,
            "recommended_aria_label_template": "예측 신뢰도 {label}, 예상 범위 {low}부터 {high}",
        },
    }


def build_markdown(copy: dict[str, object], policy: dict[str, object]) -> str:
    evaluation = policy["evaluation"]
    high = evaluation["high_risk"]
    global_row = policy["confidence_report_summary"]["global"]
    tier_summary = policy["confidence_report_summary"]["tier_summary"]

    lines = [
        "# 위험 케이스 UX / 문구 설계",
        "",
        "## 목적",
        "- 최종 가격 모델 `canonical_F18_reference_huber_010`의 예측값은 그대로 둡니다.",
        "- `F41_confidence_interval_policy`는 위험 케이스를 표시하고 예상 오차 범위를 안내하는 UX 장치로만 사용합니다.",
        "- 지도 탐색을 방해하지 않기 위해 기본 marker에는 경고를 붙이지 않고, detail drawer에서만 노출합니다.",
        "",
        "## 적용 위치",
        "| UI 단위 | 적용 | 원칙 |",
        "| --- | --- | --- |",
        "| complex marker | 기본 미노출 | marker는 가격/세대수 scan 우선 |",
        "| selected marker | 선택적 약한 border | high risk일 때만 지도 맥락 보조 |",
        "| detail drawer | 주 노출 | 예측가 바로 아래 compact row |",
        "| mobile sheet | 주 노출 | 첫 화면에는 한 줄 요약, 상세 문구는 접기 |",
        "",
        "## 위험 상태",
        "| 상태 | 조건 | chip | 기본 문구 | 보조 문구 | 구간 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for state in ["normal", "low_history", "high_risk", "unknown"]:
        item = copy["risk_states"][state]
        lines.append(
            f"| `{state}` | `{item['condition']}` | {item['label']} | {item['primary']} | {item['secondary']} | {item['interval']} |"
        )

    lines.extend(
        [
            "",
            "## Detail Drawer 구성",
            "```text",
            "예측가 8.4억",
            "[큰 오차 주의] 예상 범위 7.1억 ~ 9.8억",
            "과거 오차 패턴상 실제 가격과 크게 차이날 가능성이 평균보다 높습니다.",
            "기준: 단지 기준 p95 · 최근 실거래와 층/동/수리 상태 확인 필요",
            "```",
            "",
            "일반 상태:",
            "```text",
            "예측가 8.4억",
            "[예측 신뢰도 보통] 예상 범위 7.3억 ~ 9.5억",
            "최근 비교 이력을 기준으로 예측 범위를 산정했습니다.",
            "기준: 단지 기준 p90",
            "```",
            "",
            "## 문구 카탈로그",
            "| key | text |",
            "| --- | --- |",
            f"| section_title | {copy['component_copy']['section_title']} |",
            f"| range_label | {copy['component_copy']['range_label']} |",
            f"| source_label | {copy['component_copy']['source_label']} |",
            f"| range_helper | {copy['component_copy']['range_helper']} |",
            f"| disclaimer | {copy['component_copy']['disclaimer']} |",
            "",
            "## 구간 계산",
            "```text",
            "interval_pct = p90_abs_pct_error 또는 p95_abs_pct_error",
            "lower = predicted_price * max(0, 1 - interval_pct)",
            "upper = predicted_price * (1 + interval_pct)",
            "```",
            "",
            "기본은 p90입니다. `high_risk`, `unknown`, 또는 보수적 안내가 필요한 화면에서는 p95를 우선 표시합니다.",
            "",
            "## Fallback 기준",
            "| 우선순위 | 기준 | 표시 문구 |",
            "| ---: | --- | --- |",
            "| 1 | complex | 단지 기준 |",
            "| 2 | legal_dong | 동 기준 |",
            "| 3 | sgg | 구 기준 |",
            "| 4 | sido | 시도 기준 |",
            "| 5 | global | 전체 기준 |",
            "",
            "## 검증 수치",
            "| metric | value |",
            "| --- | ---: |",
            f"| recent_holdout rows | {evaluation['recent_holdout_rows']} |",
            f"| high-risk rows | {high['rows']} |",
            f"| high-risk coverage | {high['coverage'] * 100:.4f}% |",
            f"| high-risk >20% error rate | {high['error_gt_20pct_rate'] * 100:.4f}% |",
            f"| overall >20% error rate | {evaluation['overall_error_gt_20pct_rate'] * 100:.4f}% |",
            f"| lift vs overall | {high['lift_vs_overall']:.2f}x |",
            "",
            "## Global fallback 값",
            "| metric | value |",
            "| --- | ---: |",
            f"| source_until_ym | {global_row['source_until_ym']} |",
            f"| rows | {global_row['rows']} |",
            f"| p90 | {global_row['p90_abs_pct_error'] * 100:.4f}% |",
            f"| p95 | {global_row['p95_abs_pct_error'] * 100:.4f}% |",
            "",
            "## Tier 중앙값",
            "| tier | groups | median p90 | median p95 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in tier_summary:
        lines.append(
            f"| {row['confidence_tier']} | {row['groups']} | {row['median_p90'] * 100:.4f}% | {row['median_p95'] * 100:.4f}% |"
        )

    lines.extend(
        [
            "",
            "## UX Guardrails",
            "- 경고 문구는 가격을 덮지 않고, 가격 바로 아래 한 줄 요약으로 시작합니다.",
            "- 색만으로 상태를 전달하지 않습니다. chip text, border, icon 또는 위치 정보를 함께 씁니다.",
            "- 큰 경고 card를 추가하지 않습니다. detail drawer 안의 compact row로 유지합니다.",
            "- `ranking`, `favorite`, `alarm`, `recommendation` 흐름으로 확장하지 않습니다.",
            "- public API URL/field/unit 변경은 없습니다.",
            "",
            "## 구현 handoff",
            "- UI component candidate: `PredictionConfidenceRow` inside complex detail drawer.",
            "- Input candidate: predicted price, selected confidence report row, `resid_risk_tier`, display source level.",
            "- Test seam candidate: risk state mapper and interval formatter.",
            "- First RED candidate: high risk + p95 case renders `큰 오차 주의`, p95 interval, and source label without changing predicted price.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not POLICY_JSON.exists():
        raise SystemExit(f"missing policy json: {POLICY_JSON}")

    policy = json.loads(POLICY_JSON.read_text(encoding="utf-8"))
    copy = build_copy(policy)
    OUTPUT_JSON.write_text(json.dumps(copy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(copy, policy), encoding="utf-8")

    manifest = {
        "artifact_name": "f18_risk_case_ux_copy",
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_policy": str(POLICY_JSON),
        "source_policy_sha256": sha256_file(POLICY_JSON),
        "generated_files": [file_record(OUTPUT_MD), file_record(OUTPUT_JSON)],
        "api_contract_impact": "none",
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"locked ux copy: {OUTPUT_MD}")
    print(f"locked ux copy json: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
