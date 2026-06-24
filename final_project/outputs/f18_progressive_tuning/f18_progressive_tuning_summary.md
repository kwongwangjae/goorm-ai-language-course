# F18 Progressive Tuning Summary

- epoch: `30`
- seeds: `183,184,185`

## Recent Holdout Mean Ranking

| rank | candidate | runs | MAE_mean | MAE_std | p95_mean | p99_mean | >10%_mean | >20%_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | F36_monthly_market_anchor_huber_010 | 3 | 0.063465 | 0.000046 | 18.8562% | 35.4044% | 18.3242% | 4.3246% |
| 2 | F18_reference_huber_010 | 3 | 0.066620 | 0.000125 | 20.1446% | 36.5110% | 19.8520% | 5.0789% |

## Stage Decisions

| stage | candidate | reference | action | reason |
| --- | --- | --- | --- | --- |
| 1_baseline_repro | F18_reference_huber_010 | - | complete | baseline repeated across seeds |
| 2_monthly_anchor | F36_monthly_market_anchor_huber_010 | F18_reference_huber_010 | adopt_and_stop | mean MAE improved by 0.003155; later low-priority stages stopped by decision |
| 3_monthly_prev3 | F37_monthly_anchor_prev3_rolling_huber_010 | F36_monthly_market_anchor_huber_010 | skipped | lower expected value after prior prev3 result and strong monthly-anchor gain |
| 4_sparse_gap | F38_monthly_sparse_gap_huber_010 | F36_monthly_market_anchor_huber_010 | skipped | lower expected value; avoid extra tuning without clear signal |
