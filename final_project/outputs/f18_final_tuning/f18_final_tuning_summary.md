# F18 Final Tuning Summary

## Best

- reference: `e30_mse`
- best: `e30_huber_015`
- kind: `huber`
- epoch: `30`
- guardrail: `pass`

## Top Candidates

| rank | candidate | kind | epoch | MAE | p95 | p99 | >10% | >20% | d_MAE | d_p99 | d_gt20 | guardrail |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | e30_huber_015 | huber | 30 | 0.066424 | 20.0117% | 36.4209% | 19.7276% | 5.0060% | -0.001858 | -0.1108% | -0.1332% | pass |
| 2 | e30_huber_015_gate_resid_expected_abs_pct_error_q50 | risk_gate | 30 | 0.066427 | 19.9190% | 36.1152% | 19.7424% | 4.9568% | -0.001855 | -0.4165% | -0.1824% | pass |
| 3 | e30_huber_015_gate_best_resid_abs_pct_mean_q50 | risk_gate | 30 | 0.066427 | 19.9190% | 36.1152% | 19.7424% | 4.9568% | -0.001855 | -0.4165% | -0.1824% | pass |
| 4 | e50_huber_015 | huber | 50 | 0.066437 | 20.0161% | 36.4029% | 19.7414% | 5.0070% | -0.001845 | -0.1288% | -0.1322% | pass |
| 5 | e50_huber_015_gate_resid_expected_abs_pct_error_q50 | risk_gate | 50 | 0.066439 | 19.9286% | 36.0859% | 19.7543% | 4.9573% | -0.001842 | -0.4459% | -0.1819% | pass |
| 6 | e50_huber_015_gate_best_resid_abs_pct_mean_q50 | risk_gate | 50 | 0.066439 | 19.9286% | 36.0859% | 19.7543% | 4.9573% | -0.001842 | -0.4459% | -0.1819% | pass |
| 7 | e100_huber_015 | huber | 100 | 0.066457 | 20.0014% | 36.3924% | 19.7457% | 5.0008% | -0.001825 | -0.1393% | -0.1384% | pass |
| 8 | e30_huber_015_gate_resid_expected_abs_pct_error_q85 | risk_gate | 30 | 0.066464 | 19.9902% | 36.3510% | 19.7524% | 4.9969% | -0.001817 | -0.1808% | -0.1423% | pass |
| 9 | e30_huber_015_gate_best_resid_abs_pct_mean_q85 | risk_gate | 30 | 0.066464 | 19.9902% | 36.3510% | 19.7524% | 4.9969% | -0.001817 | -0.1808% | -0.1423% | pass |
| 10 | e100_huber_015_gate_resid_expected_abs_pct_error_q50 | risk_gate | 100 | 0.066467 | 19.9363% | 36.0584% | 19.7663% | 4.9573% | -0.001815 | -0.4733% | -0.1819% | pass |
| 11 | e100_huber_015_gate_best_resid_abs_pct_mean_q50 | risk_gate | 100 | 0.066467 | 19.9363% | 36.0584% | 19.7663% | 4.9573% | -0.001815 | -0.4733% | -0.1819% | pass |
| 12 | e30_huber_015_gate_resid_expected_abs_pct_error_q70 | risk_gate | 30 | 0.066469 | 19.9934% | 36.3857% | 19.7629% | 4.9974% | -0.001813 | -0.1460% | -0.1418% | pass |
| 13 | e30_huber_015_gate_best_resid_abs_pct_mean_q70 | risk_gate | 30 | 0.066469 | 19.9934% | 36.3857% | 19.7629% | 4.9974% | -0.001813 | -0.1460% | -0.1418% | pass |
| 14 | e50_huber_010_gate_resid_expected_abs_pct_error_q50 | risk_gate | 50 | 0.066469 | 20.1017% | 36.4839% | 19.8193% | 5.0566% | -0.001812 | -0.0478% | -0.0826% | pass |
| 15 | e50_huber_010_gate_best_resid_abs_pct_mean_q50 | risk_gate | 50 | 0.066469 | 20.1017% | 36.4839% | 19.8193% | 5.0566% | -0.001812 | -0.0478% | -0.0826% | pass |
| 16 | e30_huber_010_gate_resid_expected_abs_pct_error_q50 | risk_gate | 30 | 0.066474 | 20.0984% | 36.4435% | 19.8236% | 5.0538% | -0.001808 | -0.0882% | -0.0855% | pass |
| 17 | e30_huber_010_gate_best_resid_abs_pct_mean_q50 | risk_gate | 30 | 0.066474 | 20.0984% | 36.4435% | 19.8236% | 5.0538% | -0.001808 | -0.0882% | -0.0855% | pass |
| 18 | e50_huber_015_gate_resid_expected_abs_pct_error_q85 | risk_gate | 50 | 0.066474 | 19.9963% | 36.3564% | 19.7624% | 4.9984% | -0.001807 | -0.1754% | -0.1408% | pass |
| 19 | e50_huber_015_gate_best_resid_abs_pct_mean_q85 | risk_gate | 50 | 0.066474 | 19.9963% | 36.3564% | 19.7624% | 4.9984% | -0.001807 | -0.1754% | -0.1408% | pass |
| 20 | e50_huber_015_gate_resid_expected_abs_pct_error_q70 | risk_gate | 50 | 0.066480 | 19.9987% | 36.3819% | 19.7720% | 4.9993% | -0.001801 | -0.1498% | -0.1399% | pass |
| 21 | e50_huber_015_gate_best_resid_abs_pct_mean_q70 | risk_gate | 50 | 0.066480 | 19.9987% | 36.3819% | 19.7720% | 4.9993% | -0.001801 | -0.1498% | -0.1399% | pass |
| 22 | e30_huber_015_gate_sgg_resid_abs_pct_mean_q85 | risk_gate | 30 | 0.066481 | 20.0132% | 36.4187% | 19.7615% | 5.0084% | -0.001801 | -0.1130% | -0.1308% | pass |
| 23 | e30_huber_015_gate_sgg_resid_error_gt_20_rate_q85 | risk_gate | 30 | 0.066488 | 20.0238% | 36.4254% | 19.7610% | 5.0117% | -0.001793 | -0.1064% | -0.1275% | pass |
| 24 | e50_huber_015_gate_sgg_resid_abs_pct_mean_q85 | risk_gate | 50 | 0.066489 | 20.0168% | 36.3895% | 19.7663% | 5.0074% | -0.001792 | -0.1422% | -0.1318% | pass |
| 25 | e50_huber_015_gate_sgg_resid_error_gt_20_rate_q85 | risk_gate | 50 | 0.066496 | 20.0243% | 36.4056% | 19.7663% | 5.0108% | -0.001786 | -0.1261% | -0.1284% | pass |
