# E10 outlier-signal / robust training 최종 판단

## 결론
- outlier signal feature 추가는 채택하지 않습니다.
- `F18_exact_area_prev_additive` feature 구조는 유지합니다.
- robust loss는 평균 성능을 개선하지만 p99 tail tradeoff가 있어 조건부 채택입니다.

## 검증 요약
- sidecar: `outputs/e10_outlier_signal_features.csv`
- quality report: `outputs/e10_outlier_signal_feature_quality_report.md`
- rows: `3,593,663`
- leakage guard: 현재 거래가격 기반 outlier flag는 사용하지 않았습니다.
- `sgg` prior aggregate는 같은 `deal_ym`을 제외하고 이전 월만 사용했습니다.

## Smoke 탈락
아래 후보는 smoke에서 `F18_reference_recheck` 대비 악화되어 full 후보에서 제외했습니다.

- `F19_outlier_signal_minimal`
- `F19_outlier_signal_minimal_huber`
- `F20_outlier_signal_basic`
- `F21_outlier_signal_region_prior`
- `F22_outlier_signal_region_huber`
- `F18_reference_logcosh`

## Full 결과
현재 최종 summary는 `F18_reference_huber_010` full run 결과입니다.

| model | valid log_mae | test log_mae | recent_holdout log_mae | recent gt10 | recent gt20 | recent p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F18_reference_recheck | 0.058673 | 0.058482 | 0.062146 | 0.180304 | 0.042899 | 0.338300 |
| F18_reference_huber_010 | 0.058549 | 0.058180 | 0.061775 | 0.177473 | 0.042952 | 0.345582 |

별도 full run에서 `F18_reference_huber` (`delta=0.05`)도 확인했습니다.

| model | valid log_mae | test log_mae | recent_holdout log_mae | recent gt10 | recent gt20 | recent p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F18_reference_huber_005 | 0.058625 | 0.058153 | 0.061547 | 0.177426 | 0.042856 | 0.345716 |

## 채택 판단
- primary metric인 `recent_holdout log_mae`만 보면 `Huber(delta=0.05)`가 가장 좋습니다.
- `error_gt_10pct_rate`도 Huber 계열이 개선됩니다.
- 그러나 `abs_pct_error_p99`는 `F18_reference_recheck`보다 악화됩니다.

따라서 최종 기준은 두 가지로 나눕니다.

- 평균/10% 초과율 우선: `F18_reference_huber_005`
- p99 tail 보수 우선: `F18_reference_recheck`

모델 개발 마무리 기준으로는 `F18_reference_recheck`를 안전 baseline으로 유지하고, `Huber(delta=0.05)`는 성능 개선 후보로 기록합니다.
