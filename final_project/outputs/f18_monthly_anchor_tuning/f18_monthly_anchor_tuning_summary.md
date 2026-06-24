# F18 Monthly Anchor Tuning Summary

## Best

- reference: `F18_reference_huber_010`
- best: `F36_monthly_market_anchor_huber_010`
- epoch: `30`
- guardrail: `pass`

## Recent Holdout Ranking

| rank | candidate | MAE | p95 | p99 | >10% | >20% | d_MAE | d_p99 | d_gt20 | guardrail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | F36_monthly_market_anchor_huber_010 | 0.063460 | 18.8802% | 35.3988% | 18.3875% | 4.3400% | -0.003076 | -1.0564% | -0.7357% | pass |
| 2 | F18_reference_huber_010 | 0.066536 | 20.1510% | 36.4552% | 19.8298% | 5.0757% | 0.000000 | 0.0000% | 0.0000% | baseline |
