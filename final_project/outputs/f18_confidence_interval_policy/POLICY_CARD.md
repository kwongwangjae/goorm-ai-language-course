# F18 Confidence Interval Policy

## Final decision
- policy: `F41_confidence_interval_policy`
- status: `adopted_as_risk_policy`
- price model: `canonical_F18_reference_huber_010`
- price prediction: unchanged
- purpose: show expected error interval and confidence/risk warning.

## What this policy does
- It does not improve MAE/p95/p99 directly.
- It uses historical residual error by complex/region to show an expected percent error range.
- It prioritizes user caution when the model is likely unstable.

## Evaluation signal

| metric | value |
| --- | ---: |
| recent_holdout rows | 209468 |
| high-risk rows | 28738 |
| high-risk coverage | 13.7195% |
| high-risk >20% error rate | 8.2991% |
| overall >20% error rate | 5.1225% |
| lift vs overall | 1.62x |

## Interval lookup rule

| priority | level | key |
| ---: | --- | --- |
| 1 | complex | `complex_id` |
| 2 | legal_dong | `legal_dong_code` |
| 3 | sgg | `sgg_code` |
| 4 | sido | `sido_code` |
| 5 | global | fallback |

Use `p90_abs_pct_error` as the default interval and `p95_abs_pct_error` when a conservative range is needed.

```text
lower = predicted_price * max(0, 1 - interval_pct)
upper = predicted_price * (1 + interval_pct)
```

## Confidence tier rule

| tier | rule |
| --- | --- |
| high | rows >= 300 and p90 <= 20% |
| medium | rows >= 50 and p90 <= 30% |
| low | rows > 0 but not high/medium |
| no_history | no residual history; use fallback |

## Global fallback

| metric | value |
| --- | ---: |
| source_until_ym | 2025-12 |
| rows | 1893038 |
| p80 | 11.2199% |
| p90 | 16.3597% |
| p95 | 22.1796% |

## Tier summary

| confidence_tier | groups | median p80 | median p90 | median p95 |
| --- | ---: | ---: | ---: | ---: |
| high | 2062 | 10.1142% | 14.2278% | 18.6173% |
| low | 29328 | 16.7162% | 20.8403% | 23.9938% |
| medium | 11723 | 10.4384% | 14.2945% | 18.3038% |

## Display policy
- normal: show price plus `예상 오차 범위 ±p90`.
- low confidence or high/unknown residual risk: show price as reference and emphasize confidence warning.
- do not apply this as a correction to the predicted price.

## Source note
- candidate reason: `high-risk coverage=0.1372; high-risk >20%=0.0830; overall >20%=0.0512; lift=1.62`
- F41 lift is measured on E11 same-run eval predictions, not as a replacement for the canonical F18 price metric.
