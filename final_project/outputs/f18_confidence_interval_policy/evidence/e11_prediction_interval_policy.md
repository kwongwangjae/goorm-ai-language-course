# E11 prediction interval policy

## 1. 결론
- 가격 예측값 자체와 별개로, residual OOF 이력에서 지역/단지별 예상 오차 범위를 제공합니다.
- 기본 표기는 `예상 오차 범위: ±p90`이며, 보수적 안내가 필요하면 `±p95`를 사용합니다.
- residual source는 현재 거래월보다 이전 월까지만 사용합니다.
- confidence report 기본 source 상한은 `2025-12`입니다.

## 2. Global fallback
- source_until_ym: `2025-12`
- rows: `1,893,038`
- p80_abs_pct_error: `0.1122`
- p90_abs_pct_error: `0.1636`
- p95_abs_pct_error: `0.2218`

## 3. Tier policy
| confidence_tier | groups | median_p80 | median_p90 | median_p95 |
| --- | --- | --- | --- | --- |
| high | 2062 | 0.101142 | 0.142278 | 0.186173 |
| low | 29328 | 0.167162 | 0.208403 | 0.239938 |
| medium | 11723 | 0.104384 | 0.142945 | 0.183038 |

## 4. 사용 규칙
- `complex` 통계가 충분하면 complex 기준 p90을 우선 사용합니다.
- complex 이력이 부족하면 `legal_dong -> sgg -> sido -> global` 순서로 fallback합니다.
- `confidence_tier=low` 또는 `resid_risk_tier=high|unknown`이면 가격 대신 신뢰도 안내를 우선 노출합니다.
- confidence_report_csv: `/Users/gwongwangjae/goorm-ai-language-course/final_project/outputs/e11_region_confidence_report.csv`
