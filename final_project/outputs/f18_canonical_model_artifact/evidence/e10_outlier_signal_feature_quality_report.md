# E10 outlier-signal feature 품질 리포트

- 품질 등급: `Pass`
- rows: 3,593,663
- leakage guard: 현재 거래가격 기반 outlier flag는 feature로 저장하지 않습니다.

## 지적사항
- none

## 검증 근거 확인
- row_count_match: pass
- transaction_id_unique: pass
- join_missing_zero: pass
- sgg_lag1_source_before_deal_ym: pass
- numeric_features_finite_or_null: pass

## Coverage
| metric | value |
| --- | --- |
| rows | 3593663.000000 |
| sgg_prior_missing | 28878.000000 |
| sgg_prior_missing_rate | 0.008036 |
| wide_prev_jump_20pct_rate | 0.089218 |
| exact_prev_jump_20pct_rate | 0.088241 |
| exact_wide_gap_10pct_rate | 0.025487 |
| wide_region_outlier_20pct_rate | 0.617255 |
| exact_region_outlier_20pct_rate | 0.614726 |

## 검증 공백
- `sgg` prior feature는 같은 `deal_ym`을 제외하고 이전 월 aggregate만 사용합니다.
- prev jump/exact-wide gap feature는 이미 생성된 과거 거래 feature만 사용합니다.
- sidecar_csv: `/Users/gwongwangjae/goorm-ai-language-course/final_project/outputs/e10_outlier_signal_features.csv`
