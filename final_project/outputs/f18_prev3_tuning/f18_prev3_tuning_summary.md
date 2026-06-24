# F18 Prev3 Rolling Tuning Summary

## Best

- reference: `F18_reference_huber_010`
- best: `F35_prev3_rolling_huber_010`
- epoch: `30`
- guardrail: `pass`

## Recent Holdout Ranking

| rank | candidate | MAE | p95 | p99 | >10% | >20% | d_MAE | d_p99 | d_gt20 | guardrail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | F35_prev3_rolling_huber_010 | 0.064543 | 19.5249% | 36.1046% | 18.9346% | 4.7205% | -0.002055 | -0.7710% | -0.4067% | pass |
| 2 | F18_reference_huber_010 | 0.066597 | 20.2273% | 36.8757% | 19.8589% | 5.1273% | 0.000000 | 0.0000% | 0.0000% | baseline |
