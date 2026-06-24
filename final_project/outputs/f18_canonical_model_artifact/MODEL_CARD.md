# F18 Canonical Model Artifact

## Final decision
- final price model: `canonical_F18_reference_huber_010`
- source experiment: `F18_reference_huber_010`
- training epoch: `30`
- loss: `Huber(delta=0.10)`
- price correction: not adopted
- confidence/risk policy: keep separate from price prediction
- trained model weights: not present in the current saved artifacts; this freeze locks the reproducible model definition and evaluation evidence.

## Locked recent_holdout metrics

| metric | value |
| --- | ---: |
| MAE(log) | 0.061775 |
| p95 | 18.8077% |
| p99 | 34.5582% |
| >10% | 17.7473% |
| >20% | 4.2952% |

## Split metrics

| split | MAE(log) | p95 | p99 | >10% | >20% | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.058549 | 17.6666% | 31.0423% | 16.7366% | 3.6381% | 384432 |
| test | 0.058180 | 17.8412% | 32.0412% | 16.3431% | 3.7778% | 442008 |
| recent_holdout | 0.061775 | 18.8077% | 34.5582% | 17.7473% | 4.2952% | 209468 |

## Artifact contents
- `manifest.json`: locked source paths, file sizes, and SHA-256 hashes
- `feature_schema.json`: final feature contract
- `final_metrics.csv`: locked final model metrics
- `evidence/`: copied E10 evaluation reports used for the decision

## Reproduction command
```bash
cd /Users/gwongwangjae/goorm-ai-language-course/final_project
E10_RUN_MODE=full E10_EXPERIMENTS=F18_reference_recheck,F18_reference_huber_010 E10_MAX_EPOCHS=30 python scripts/run_e10_outlier_signal_experiments.py
```
