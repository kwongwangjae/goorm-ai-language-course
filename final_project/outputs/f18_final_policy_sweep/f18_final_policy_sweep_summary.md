# F18 Final Policy Sweep Summary

## Canonical Reference

| candidate | MAE | p95 | p99 | >10% | >20% |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical_F18_reference_huber_010 | 0.061775 | 18.8077% | 34.5582% | 17.7473% | 4.2952% |

## Candidate Results

| order | candidate | kind | status | MAE | p95 | p99 | >10% | >20% / risk >20% | lift/delta | reason |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | F40_sparse_fallback_policy | price_policy | discarded | 0.067780 | 20.1326% | 36.2797% | 20.2828% | 5.0767% | -0.000426 | sparse rows=7467; same-run d_MAE=-0.000426, d_p99=-0.002099, d_gt20=-0.000458; beats_same_run=True, beats_canonical=False |
| 2 | F41_confidence_interval_policy | risk_policy | candidate |  |  |  |  | 8.2991% | 1.620130 | high-risk coverage=0.1372; high-risk >20%=0.0830; overall >20%=0.0512; lift=1.62 |
| 3 | F42_monthly_anchor_gated | price_policy | discarded |  |  |  |  |  |  | long-gap groups with MAE gain=5/5, p99 worse=4/5; full monthly did not beat canonical |
| 4 | F43_reconstruction_news_risk_only | risk_policy | discarded |  |  |  |  | 0.0000% | 0.000000 | threshold=0.8783; flagged coverage=0.0000; flagged >20%=0.0000; overall >20%=0.0512; lift=0.00 |

## Final

- price model: `canonical_F18_reference_huber_010`
- adopted risk policies: `F41_confidence_interval_policy`
- discarded price policies: `F40_sparse_fallback_policy`, `F42_monthly_anchor_gated`
