#!/usr/bin/env python3
"""Regression tests for the E11 epoch sweep orchestrator."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from run_e11_epoch_sweep import (
    BASELINE_CANDIDATE,
    PRICE_CANDIDATES,
    RUN_CANDIDATES,
    SUPPORT_POLICY_CANDIDATE,
    build_selection_rows,
    collect_epoch_metrics,
)


class E11EpochSweepTest(unittest.TestCase):
    def test_collect_epoch_metrics_reads_e11_long_format_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            epoch_dir = Path(temp_dir) / "epoch_10"
            epoch_dir.mkdir(parents=True)
            metrics_path = epoch_dir / "e11_region_residual_metrics.csv"
            with metrics_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "run_mode",
                        "experiment_name",
                        "split",
                        "rows",
                        "log_mae",
                        "abs_pct_error_p99",
                        "error_gt_20pct_rate",
                    ],
                )
                writer.writeheader()
                for candidate in RUN_CANDIDATES:
                    for split in ("valid", "test", "recent_holdout"):
                        writer.writerow(
                            {
                                "run_mode": "full",
                                "experiment_name": candidate,
                                "split": split,
                                "rows": "1",
                                "log_mae": {
                                    "valid": "0.100",
                                    "test": "0.110",
                                    "recent_holdout": "0.120",
                                }[split],
                                "abs_pct_error_p99": "0.300",
                                "error_gt_20pct_rate": "0.050",
                            }
                        )
            (epoch_dir / "e11_region_residual_feature_quality_report.md").write_text(
                "# Quality\n\n- 품질 등급: `Pass`\n",
                encoding="utf-8",
            )

            results = collect_epoch_metrics(10, epoch_dir)

            self.assertEqual(set(results), set(RUN_CANDIDATES))
            self.assertEqual(results[BASELINE_CANDIDATE].valid_log_mae, 0.100)
            self.assertEqual(results[PRICE_CANDIDATES[0]].test_log_mae, 0.110)
            self.assertEqual(results[PRICE_CANDIDATES[1]].recent_holdout_abs_pct_error_p99, 0.300)
            self.assertEqual(results[PRICE_CANDIDATES[2]].recent_holdout_error_gt_20pct_rate, 0.050)
            self.assertEqual(results[SUPPORT_POLICY_CANDIDATE].residual_quality_status, "Pass")

    def test_collect_epoch_metrics_accepts_filtered_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            epoch_dir = Path(temp_dir) / "epoch_100"
            epoch_dir.mkdir(parents=True)
            metrics_path = epoch_dir / "e11_region_residual_metrics.csv"
            with metrics_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "run_mode",
                        "experiment_name",
                        "split",
                        "rows",
                        "log_mae",
                        "abs_pct_error_p99",
                        "error_gt_20pct_rate",
                    ],
                )
                writer.writeheader()
                for candidate, valid_log_mae in (
                    (BASELINE_CANDIDATE, "0.065"),
                    ("F29_residual_bias_features_huber", "0.064"),
                ):
                    for split in ("valid", "test", "recent_holdout"):
                        writer.writerow(
                            {
                                "run_mode": "full",
                                "experiment_name": candidate,
                                "split": split,
                                "rows": "1",
                                "log_mae": valid_log_mae if split == "valid" else "0.066",
                                "abs_pct_error_p99": "0.350",
                                "error_gt_20pct_rate": "0.050",
                            }
                        )
            (epoch_dir / "e11_region_residual_feature_quality_report.md").write_text(
                "# Quality\n\n- 품질 등급: `Pass`\n",
                encoding="utf-8",
            )

            expected = (BASELINE_CANDIDATE, "F29_residual_bias_features_huber")
            results = collect_epoch_metrics(100, epoch_dir, expected)
            rows = build_selection_rows({100: results})

            self.assertEqual(set(results), set(expected))
            self.assertEqual([row.metrics.candidate for row in rows], list(expected))
            self.assertEqual(rows[1].guardrail_status, "pass")

    def test_sweep_runs_fake_runner_preserves_outputs_and_skips_successful_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            canonical_dir = root / "canonical"
            sweep_dir = root / "sweep"
            fake_runner = root / "fake_runner.py"
            call_log = root / "calls.csv"
            fake_runner.write_text(
                textwrap.dedent(
                    f"""
                    import csv
                    import os
                    from pathlib import Path

                    out = Path(os.environ["E11_OUTPUT_DIR"])
                    out.mkdir(parents=True, exist_ok=True)
                    epoch = int(os.environ["E11_MAX_EPOCHS"])
                    candidate = os.environ.get("E11_CANDIDATE", "RESIDUAL")
                    with Path({str(call_log)!r}).open("a", encoding="utf-8", newline="") as fh:
                        csv.writer(fh).writerow([
                            epoch,
                            candidate,
                            os.environ.get("E11_REBUILD_OOF", ""),
                            os.environ.get("E11_REBUILD_RESIDUAL_FEATURES", ""),
                        ])

                    metrics = {{
                        "F18_reference_recheck": (0.1300, 0.1400, 0.3000, 0.0500),
                        "F25_sgg_bias_calibration": (0.1200, 0.1390, 0.2900, 0.0490),
                        "F26_multilevel_bias_calibration": (0.1210, 0.1390, 0.2950, 0.0495),
                        "F29_residual_bias_features_huber": (0.1190, 0.1450, 0.2900, 0.0490),
                        "F30_confidence_only_policy": (0.1320, 0.1410, 0.3010, 0.0510),
                    }}
                    if candidate == "RESIDUAL":
                        (out / "e11_oof_predictions.csv").write_text("id,prediction\\n1,10\\n", encoding="utf-8")
                        (out / "e11_residual_sidecar.csv").write_text("id,residual\\n1,0.1\\n", encoding="utf-8")
                        (out / "e11_residual_quality_report.md").write_text("# Quality\\n\\nStatus: Pass\\n", encoding="utf-8")
                    else:
                        valid, test, p99, gt20 = metrics[candidate]
                        metrics_path = out / "e11_metrics.csv"
                        exists = metrics_path.exists()
                        with metrics_path.open("a", encoding="utf-8", newline="") as fh:
                            writer = csv.DictWriter(
                                fh,
                                fieldnames=[
                                    "candidate",
                                    "valid_log_mae",
                                    "test_log_mae",
                                    "recent_holdout_abs_pct_error_p99",
                                    "recent_holdout_error_gt_20pct_rate",
                                ],
                            )
                            if not exists:
                                writer.writeheader()
                            writer.writerow({{
                                "candidate": candidate,
                                "valid_log_mae": valid,
                                "test_log_mae": test,
                                "recent_holdout_abs_pct_error_p99": p99,
                                "recent_holdout_error_gt_20pct_rate": gt20,
                            }})
                    """
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": "scripts",
                    "E11_SWEEP_EPOCHS": "2",
                    "E11_SWEEP_OUTPUT_DIR": str(sweep_dir),
                    "E11_SWEEP_CANONICAL_OUTPUT_DIR": str(canonical_dir),
                    "E11_SWEEP_RUN_COMMAND": f"{sys.executable} {fake_runner}",
                }
            )
            first = subprocess.run(
                [sys.executable, "-m", "run_e11_epoch_sweep"],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue((sweep_dir / "epoch_2" / "_SUCCESS").exists())
            self.assertTrue((sweep_dir / "epoch_2" / "e11_oof_predictions.csv").exists())
            self.assertTrue((sweep_dir / "epoch_2" / "e11_residual_sidecar.csv").exists())
            self.assertTrue((sweep_dir / "e11_epoch_sweep_metrics.csv").exists())
            decision = (sweep_dir / "e11_epoch_sweep_final_decision.md").read_text(encoding="utf-8")
            self.assertIn("F25_sgg_bias_calibration", decision)
            self.assertIn("guardrail", decision)

            with call_log.open(encoding="utf-8", newline="") as fh:
                first_calls = list(csv.reader(fh))
            self.assertEqual(len(first_calls), 6)
            self.assertEqual(first_calls[0], ["2", "RESIDUAL", "1", "1"])

            second = subprocess.run(
                [sys.executable, "-m", "run_e11_epoch_sweep"],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            with call_log.open(encoding="utf-8", newline="") as fh:
                second_calls = list(csv.reader(fh))
            self.assertEqual(second_calls, first_calls)


if __name__ == "__main__":
    unittest.main()
