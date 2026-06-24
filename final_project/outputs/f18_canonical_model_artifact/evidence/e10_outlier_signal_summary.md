# E10 outlier-signal feature 실험 요약

## 1. 결론
- 결론: `성공`
- 비교 기준은 동일 실행의 `F18_reference_recheck`입니다.
- 채택 기준: `recent_holdout log_mae` 개선, valid/test 악화 제한, recent p99 및 gt20 tail 악화 제한.

| experiment_name | valid_delta_vs_f18_recheck | test_delta_vs_f18_recheck | recent_delta_vs_f18_recheck | recent_p99_abs_pct_delta | recent_gt20_rate_delta | judgement |
| --- | --- | --- | --- | --- | --- | --- |
| F18_reference_huber_010 | -0.000124 | -0.000302 | -0.000371 | 0.007282 | 0.000053 | 성공 |

## 2. 실행 설정
- run_mode: `full`
- max_epochs: `30`
- batch_size: `8192`
- split: `train<=2023`, `valid=2024`, `test=2025`, `recent_holdout>=2026`
- Policy B: `is_cancelled == 0`, `trade_type in [중개거래, unknown]`
- leakage guard: 현재 거래가격 기반 outlier flag는 사용하지 않습니다.

## 3. Split row 수와 outlier-signal coverage
| split | rows | wide_prev_jump_20pct_rate | exact_prev_jump_20pct_rate | exact_wide_gap_10pct_rate | wide_region_outlier_20pct_rate | sgg_prior_missing_rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2342081 | 0.087232 | 0.086121 | 0.025789 | 0.600221 | 0.011575 |
| valid | 384432 | 0.081936 | 0.081981 | 0.022337 | 0.628036 | 0.000000 |
| test | 442008 | 0.078014 | 0.077750 | 0.021943 | 0.646414 | 0.000000 |
| recent_holdout | 209468 | 0.086214 | 0.085330 | 0.023507 | 0.664436 | 0.000000 |

## 4. 핵심 log_mae
| experiment_name | recent_holdout | test | valid |
| --- | --- | --- | --- |
| F18_reference_huber_010 | 0.061775 | 0.058180 | 0.058549 |
| F18_reference_recheck | 0.062146 | 0.058482 | 0.058673 |

## 5. Tail metrics
| experiment_name | split | abs_pct_error_p95 | abs_pct_error_p99 | error_gt_10pct_rate | error_gt_20pct_rate | delta_vs_f18_recheck |
| --- | --- | --- | --- | --- | --- | --- |
| F18_reference_recheck | valid | 0.176751 | 0.305136 | 0.168865 | 0.035720 | 0.000000 |
| F18_reference_recheck | test | 0.178389 | 0.313676 | 0.165198 | 0.037420 | 0.000000 |
| F18_reference_recheck | recent_holdout | 0.188072 | 0.338300 | 0.180304 | 0.042899 | 0.000000 |
| F18_reference_huber_010 | valid | 0.176666 | 0.310423 | 0.167366 | 0.036381 | -0.000124 |
| F18_reference_huber_010 | test | 0.178412 | 0.320412 | 0.163431 | 0.037778 | -0.000302 |
| F18_reference_huber_010 | recent_holdout | 0.188077 | 0.345582 | 0.177473 | 0.042952 | -0.000371 |

## 6. Focus group metrics
| experiment_name | split | group_type | group_value | rows | log_mae | p95_abs_pct_error | p99_abs_pct_error | error_gt_20pct_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F18_reference_recheck | recent_holdout | exact_prev1_gap_bucket_plus | 366-730 | 3303 | 0.117704 | 0.331630 | 0.590486 | 0.165910 |
| F18_reference_recheck | recent_holdout | exact_prev1_gap_bucket_plus | 731+ | 3281 | 0.147253 | 0.469555 | 0.781201 | 0.274611 |
| F18_reference_recheck | recent_holdout | prev1_gap_bucket_plus | 366-730 | 2677 | 0.124845 | 0.348149 | 0.616986 | 0.184535 |
| F18_reference_recheck | recent_holdout | prev1_gap_bucket_plus | 731+ | 2322 | 0.163637 | 0.517494 | 0.833790 | 0.319121 |
| F18_reference_recheck | recent_holdout | prev2_gap_bucket_plus | 366-730 | 5741 | 0.103043 | 0.289219 | 0.491850 | 0.127852 |
| F18_reference_recheck | recent_holdout | prev2_gap_bucket_plus | 731+ | 5009 | 0.131516 | 0.413196 | 0.701560 | 0.217409 |
| F18_reference_recheck | recent_holdout | wide_prev_jump_20_group | 1 | 18059 | 0.111132 | 0.301381 | 0.510550 | 0.144083 |
| F18_reference_recheck | recent_holdout | exact_prev_jump_20_group | 1 | 17874 | 0.110038 | 0.300995 | 0.512804 | 0.142777 |
| F18_reference_recheck | recent_holdout | exact_wide_gap_10_group | 1 | 4924 | 0.097083 | 0.290886 | 0.531543 | 0.116166 |
| F18_reference_recheck | recent_holdout | wide_region_outlier_20_group | 1 | 139178 | 0.064351 | 0.197728 | 0.347768 | 0.048513 |
| F18_reference_huber_010 | recent_holdout | exact_prev1_gap_bucket_plus | 366-730 | 3303 | 0.117443 | 0.336104 | 0.613358 | 0.165304 |
| F18_reference_huber_010 | recent_holdout | exact_prev1_gap_bucket_plus | 731+ | 3281 | 0.144505 | 0.466119 | 0.780106 | 0.256020 |
| F18_reference_huber_010 | recent_holdout | prev1_gap_bucket_plus | 366-730 | 2677 | 0.124495 | 0.348006 | 0.626267 | 0.175943 |
| F18_reference_huber_010 | recent_holdout | prev1_gap_bucket_plus | 731+ | 2322 | 0.159668 | 0.500594 | 0.822509 | 0.296296 |
| F18_reference_huber_010 | recent_holdout | prev2_gap_bucket_plus | 366-730 | 5741 | 0.102795 | 0.288579 | 0.524652 | 0.130639 |
| F18_reference_huber_010 | recent_holdout | prev2_gap_bucket_plus | 731+ | 5009 | 0.129624 | 0.416985 | 0.736958 | 0.206029 |
| F18_reference_huber_010 | recent_holdout | wide_prev_jump_20_group | 1 | 18059 | 0.110289 | 0.310700 | 0.548009 | 0.146520 |
| F18_reference_huber_010 | recent_holdout | exact_prev_jump_20_group | 1 | 17874 | 0.109202 | 0.309684 | 0.547038 | 0.144847 |
| F18_reference_huber_010 | recent_holdout | exact_wide_gap_10_group | 1 | 4924 | 0.097319 | 0.302425 | 0.557369 | 0.118603 |
| F18_reference_huber_010 | recent_holdout | wide_region_outlier_20_group | 1 | 139178 | 0.063879 | 0.197305 | 0.356755 | 0.048542 |

## 7. 생성 산출물
- `/Users/gwongwangjae/goorm-ai-language-course/final_project/outputs/e10_outlier_signal_features.csv`
- `/Users/gwongwangjae/goorm-ai-language-course/final_project/outputs/e10_outlier_signal_metrics.csv`
- `/Users/gwongwangjae/goorm-ai-language-course/final_project/outputs/e10_outlier_signal_group_metrics.csv`
- `/Users/gwongwangjae/goorm-ai-language-course/final_project/outputs/e10_outlier_signal_summary.md`
