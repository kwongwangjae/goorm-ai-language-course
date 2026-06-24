# F18 Floor Tuning Summary

## Best

- reference: `F18_reference_huber_010`
- best: `F34_floor_full_huber_010`
- epoch: `30`
- guardrail: `pass`

## Recent Holdout Ranking

| rank | candidate | MAE | p95 | p99 | >10% | >20% | d_MAE | d_p99 | d_gt20 | guardrail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | F34_floor_full_huber_010 | 0.065005 | 19.7554% | 36.3977% | 18.9303% | 4.8585% | -0.001529 | -0.0655% | -0.2186% | pass |
| 2 | F33_floor_interactions_huber_010 | 0.065154 | 19.8183% | 36.3924% | 19.0564% | 4.8857% | -0.001380 | -0.0708% | -0.1914% | pass |
| 3 | F32_floor_bucket_huber_010 | 0.065323 | 19.8519% | 36.4893% | 19.1504% | 4.9177% | -0.001211 | 0.0261% | -0.1595% | pass |
| 4 | F31_floor_flags_huber_010 | 0.065454 | 19.8459% | 36.2130% | 19.2177% | 4.9191% | -0.001080 | -0.2502% | -0.1580% | pass |
| 5 | F18_reference_huber_010 | 0.066534 | 20.1536% | 36.4632% | 19.8278% | 5.0771% | 0.000000 | 0.0000% | 0.0000% | baseline |
