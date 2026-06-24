# F18 Progressive Tuning Decision

- best candidate: `F36_monthly_market_anchor_huber_010`
- epoch: `30`
- seeds: `183,184,185`
- recent_holdout mean log_mae: `0.063465`
- recent_holdout mean p95: `18.8562%`
- recent_holdout mean p99: `35.4044%`
- recent_holdout mean >10%: `18.3242%`
- recent_holdout mean >20%: `4.3246%`

## Decisions
- 1_baseline_repro: `complete` - baseline repeated across seeds
- 2_monthly_anchor: `adopt_and_stop` - mean MAE improved by 0.003155; later low-priority stages stopped by decision
- 3_monthly_prev3: `skipped` - lower expected value after prior prev3 result and strong monthly-anchor gain
- 4_sparse_gap: `skipped` - lower expected value; avoid extra tuning without clear signal
