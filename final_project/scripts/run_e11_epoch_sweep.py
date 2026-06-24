#!/usr/bin/env python3
"""Run E11 full epoch sweeps and summarize candidate selection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE_CANDIDATE = "F18_reference_recheck"
PRICE_CANDIDATES = (
    "F25_sgg_bias_calibration",
    "F26_multilevel_bias_calibration",
    "F29_residual_bias_features_huber",
)
SUPPORT_POLICY_CANDIDATE = "F30_confidence_only_policy"
RUN_CANDIDATES = (BASELINE_CANDIDATE, *PRICE_CANDIDATES, SUPPORT_POLICY_CANDIDATE)
SIMPLE_PRIORITY = {candidate: index for index, candidate in enumerate(PRICE_CANDIDATES)}

DEFAULT_EPOCHS = "10,20,30,50"
DEFAULT_SWEEP_OUTPUT_DIR = REPO_ROOT / "outputs" / "e11_epoch_sweep"
DEFAULT_CANONICAL_OUTPUT_DIR = REPO_ROOT / "outputs" / "e11"

METRICS_CSV = "e11_epoch_sweep_metrics.csv"
SUMMARY_MD = "e11_epoch_sweep_summary.md"
FINAL_DECISION_MD = "e11_epoch_sweep_final_decision.md"

CSV_CANDIDATE_FIELDS = (
    "candidate",
    "experiment",
    "experiment_name",
    "experiment_id",
    "run",
    "run_id",
    "model",
    "variant",
    "feature_set",
)
SPLIT_FIELDS = ("split", "dataset", "eval_split", "holdout", "group")
METRIC_NAME_FIELDS = ("metric", "metric_name", "name")
VALUE_FIELDS = ("value", "score", "metric_value")


@dataclass(frozen=True)
class SweepConfig:
    epochs: tuple[int, ...]
    sweep_output_dir: Path
    canonical_output_dir: Path
    runner_cwd: Path
    runner_command: str | None
    residual_command: str | None
    candidate_command: str | None
    experiment_filter: str
    copy_globs: tuple[str, ...]
    force: bool
    single_run_per_epoch: bool

    @classmethod
    def from_env(cls) -> "SweepConfig":
        return cls(
            epochs=parse_epochs(os.environ.get("E11_SWEEP_EPOCHS", DEFAULT_EPOCHS)),
            sweep_output_dir=Path(os.environ.get("E11_SWEEP_OUTPUT_DIR", str(DEFAULT_SWEEP_OUTPUT_DIR))),
            canonical_output_dir=Path(
                os.environ.get("E11_SWEEP_CANONICAL_OUTPUT_DIR", str(DEFAULT_CANONICAL_OUTPUT_DIR))
            ),
            runner_cwd=Path(os.environ.get("E11_SWEEP_RUN_CWD", str(REPO_ROOT))),
            runner_command=os.environ.get("E11_SWEEP_RUN_COMMAND") or discover_default_runner_command(),
            residual_command=os.environ.get("E11_SWEEP_RESIDUAL_COMMAND"),
            candidate_command=os.environ.get("E11_SWEEP_CANDIDATE_COMMAND"),
            experiment_filter=os.environ.get("E11_SWEEP_EXPERIMENTS", ",".join(RUN_CANDIDATES)),
            copy_globs=parse_copy_globs(os.environ.get("E11_SWEEP_COPY_GLOBS", "*")),
            force=os.environ.get("E11_SWEEP_FORCE") == "1",
            single_run_per_epoch=os.environ.get("E11_SWEEP_SINGLE_RUN_PER_EPOCH") == "1",
        )


@dataclass
class CandidateMetrics:
    epoch: int
    candidate: str
    valid_log_mae: float | None = None
    test_log_mae: float | None = None
    recent_holdout_abs_pct_error_p99: float | None = None
    recent_holdout_error_gt_20pct_rate: float | None = None
    source: str = ""
    residual_quality_status: str = "Unknown"

    def has_required_metrics(self) -> bool:
        return all(
            value is not None
            for value in (
                self.valid_log_mae,
                self.test_log_mae,
                self.recent_holdout_abs_pct_error_p99,
                self.recent_holdout_error_gt_20pct_rate,
            )
        )


@dataclass(frozen=True)
class SelectionRow:
    metrics: CandidateMetrics
    delta_valid_log_mae: float | None
    delta_test_log_mae: float | None
    delta_recent_holdout_abs_pct_error_p99: float | None
    delta_recent_holdout_error_gt_20pct_rate: float | None
    guardrail_status: str
    guardrail_reasons: tuple[str, ...]
    ranking_role: str


def parse_epochs(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            epoch = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid E11_SWEEP_EPOCHS token: {token}") from exc
        if epoch <= 0:
            raise ValueError(f"epoch must be positive: {epoch}")
        values.append(epoch)
    if not values:
        raise ValueError("E11_SWEEP_EPOCHS must contain at least one epoch")
    return tuple(values)


def parse_copy_globs(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    return values or ("*",)


def discover_default_runner_command() -> str | None:
    for module in (
        "run_e11_full_experiment",
        "run_e11_experiment",
        "run_e11_pipeline",
        "run_e11",
    ):
        if (REPO_ROOT / "scripts" / f"{module}.py").exists():
            return f"{sys.executable} -m {module}"
    return None


def resolve_under_repo(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def ensure_safe_generated_dir(path: Path) -> None:
    resolved = path.resolve(strict=False)
    repo = REPO_ROOT.resolve(strict=False)
    repo_outputs = (repo / "outputs").resolve(strict=False)
    temp = Path(tempfile.gettempdir()).resolve(strict=False)
    allowed = is_relative_to(resolved, repo_outputs) or is_relative_to(resolved, temp)
    if not allowed:
        raise ValueError(f"refusing to clean generated directory outside repo outputs/temp: {resolved}")
    if resolved in {repo, repo_outputs, repo.parent, temp, temp.parent, Path("/")}:
        raise ValueError(f"refusing to clean unsafe generated directory: {resolved}")


def can_reset_generated_dir(path: Path) -> bool:
    try:
        ensure_safe_generated_dir(path)
    except ValueError:
        return False
    return True


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def reset_dir(path: Path) -> None:
    ensure_safe_generated_dir(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_epoch_dir(epoch_dir: Path, force: bool) -> None:
    if epoch_dir.exists() and force:
        reset_dir(epoch_dir)
    else:
        epoch_dir.mkdir(parents=True, exist_ok=True)


def command_for_stage(config: SweepConfig, stage: str) -> str:
    command = config.residual_command if stage == "residual" else config.candidate_command
    command = command or config.runner_command
    if not command:
        raise RuntimeError(
            "E11 runner command is not available. Set E11_SWEEP_RUN_COMMAND, "
            "E11_SWEEP_RESIDUAL_COMMAND, or E11_SWEEP_CANDIDATE_COMMAND."
        )
    return command


def expected_candidates(config: SweepConfig) -> tuple[str, ...]:
    requested = tuple(name.strip() for name in config.experiment_filter.split(",") if name.strip())
    if not requested:
        return RUN_CANDIDATES
    known = set(RUN_CANDIDATES)
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"unknown E11_SWEEP_EXPERIMENTS values: {', '.join(unknown)}")
    if BASELINE_CANDIDATE not in requested:
        raise ValueError(f"E11_SWEEP_EXPERIMENTS must include {BASELINE_CANDIDATE}")
    return requested


def format_command(command: str, *, epoch: int, stage: str, candidate: str) -> list[str]:
    formatted = command
    replacements = {
        "{python}": shlex.quote(sys.executable),
        "{epoch}": str(epoch),
        "{stage}": shlex.quote(stage),
        "{candidate}": shlex.quote(candidate),
    }
    for placeholder, value in replacements.items():
        formatted = formatted.replace(placeholder, value)
    return shlex.split(formatted)


def run_stage(
    config: SweepConfig,
    *,
    epoch: int,
    stage: str,
    candidate: str,
    epoch_dir: Path,
) -> None:
    command = command_for_stage(config, stage)
    args = format_command(command, epoch=epoch, stage=stage, candidate=candidate)
    log_path = epoch_dir / f"_run_{stage}_{candidate}.log"
    env = os.environ.copy()
    env.update(
        {
            "E11_RUN_MODE": "full",
            "E11_MAX_EPOCHS": str(epoch),
            "E11_OUTPUT_DIR": str(config.canonical_output_dir),
            "E11_SWEEP_EPOCH": str(epoch),
            "E11_SWEEP_STAGE": stage,
            "E11_CANDIDATE": candidate,
            "E11_EXPERIMENT_ID": candidate,
            "E11_VARIANT": candidate,
            "E11_FEATURE_SET": candidate,
            "E11_EXPERIMENTS": config.experiment_filter,
            "E11_REBUILD_OOF": "1" if stage in {"residual", "full"} else "0",
            "E11_REBUILD_RESIDUAL_FEATURES": "1" if stage in {"residual", "full"} else "0",
        }
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(shlex.quote(part) for part in args)}\n")
        log_file.write(f"E11_MAX_EPOCHS={epoch}\n")
        log_file.write(f"E11_CANDIDATE={candidate}\n")
        log_file.flush()
        result = subprocess.run(
            args,
            cwd=config.runner_cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{stage} run failed for epoch {epoch}, candidate {candidate}; see {log_path}")


def copy_outputs(source_dir: Path, epoch_dir: Path, copy_globs: tuple[str, ...]) -> None:
    if not source_dir.exists():
        raise RuntimeError(f"canonical E11 output directory was not created: {source_dir}")
    source = source_dir.resolve(strict=False)
    target = epoch_dir.resolve(strict=False)
    if source == target or is_relative_to(source, target):
        return
    children: list[Path] = []
    for pattern in copy_globs:
        children.extend(sorted(source.glob(pattern)))
    for child in sorted(set(children)):
        destination = epoch_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def mark_success(epoch_dir: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    (epoch_dir / "_SUCCESS").write_text(f"completed_at_utc={timestamp}\n", encoding="utf-8")


def run_epoch(config: SweepConfig, epoch: int) -> None:
    epoch_dir = config.sweep_output_dir / f"epoch_{epoch}"
    success_path = epoch_dir / "_SUCCESS"
    if success_path.exists() and not config.force:
        print(f"skip epoch {epoch}: _SUCCESS exists", flush=True)
        return

    prepare_epoch_dir(epoch_dir, config.force)
    if can_reset_generated_dir(config.canonical_output_dir):
        reset_dir(config.canonical_output_dir)
    elif not config.canonical_output_dir.exists():
        config.canonical_output_dir.mkdir(parents=True, exist_ok=True)
    if config.single_run_per_epoch:
        print(f"run epoch {epoch}: full E11 experiment set", flush=True)
        run_stage(config, epoch=epoch, stage="full", candidate="ALL", epoch_dir=epoch_dir)
        copy_outputs(config.canonical_output_dir, epoch_dir, config.copy_globs)
        collect_epoch_metrics(epoch, epoch_dir)
        mark_success(epoch_dir)
        return

    print(f"run epoch {epoch}: residual OOF/sidecar rebuild", flush=True)
    run_stage(config, epoch=epoch, stage="residual", candidate="RESIDUAL", epoch_dir=epoch_dir)
    for candidate in RUN_CANDIDATES:
        print(f"run epoch {epoch}: {candidate}", flush=True)
        run_stage(config, epoch=epoch, stage="candidate", candidate=candidate, epoch_dir=epoch_dir)
    copy_outputs(config.canonical_output_dir, epoch_dir, config.copy_globs)
    collect_epoch_metrics(epoch, epoch_dir)
    mark_success(epoch_dir)


def normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace(".", "_").replace("/", "_")


def candidate_from_text(text: str) -> str | None:
    for candidate in RUN_CANDIDATES:
        if candidate in text:
            return candidate
    return None


def candidate_from_row(row: dict[str, str], source_path: Path) -> str | None:
    for field in CSV_CANDIDATE_FIELDS:
        value = row.get(field) or row.get(normalize_key(field))
        if value:
            candidate = candidate_from_text(value)
            if candidate:
                return candidate
    return candidate_from_text(source_path.as_posix())


def first_float(row: dict[str, str], names: Iterable[str]) -> float | None:
    normalized = {normalize_key(key): value for key, value in row.items() if key is not None}
    for name in names:
        value = normalized.get(normalize_key(name))
        parsed = parse_float(value)
        if parsed is not None:
            return parsed
    return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_value(row: dict[str, str]) -> str:
    for field in SPLIT_FIELDS:
        value = row.get(field) or row.get(normalize_key(field))
        if value:
            return normalize_key(value)
    return ""


def metric_name_value(row: dict[str, str]) -> tuple[str, float | None]:
    metric_name = ""
    for field in METRIC_NAME_FIELDS:
        value = row.get(field) or row.get(normalize_key(field))
        if value:
            metric_name = normalize_key(value)
            break
    metric_value = first_float(row, VALUE_FIELDS)
    return metric_name, metric_value


def update_metric_value(metrics: CandidateMetrics, key: str, value: float | None, source: Path) -> None:
    if value is None:
        return
    if key == "valid_log_mae":
        metrics.valid_log_mae = value
    elif key == "test_log_mae":
        metrics.test_log_mae = value
    elif key == "recent_holdout_abs_pct_error_p99":
        metrics.recent_holdout_abs_pct_error_p99 = value
    elif key == "recent_holdout_error_gt_20pct_rate":
        metrics.recent_holdout_error_gt_20pct_rate = value
    if metrics.source:
        if source.as_posix() not in metrics.source:
            metrics.source = f"{metrics.source};{source.as_posix()}"
    else:
        metrics.source = source.as_posix()


def update_metrics_from_row(metrics: CandidateMetrics, row: dict[str, str], source: Path) -> None:
    update_metric_value(
        metrics,
        "valid_log_mae",
        first_float(row, ("valid_log_mae", "validation_log_mae", "valid_mae_log", "valid_log_error_mae")),
        source,
    )
    update_metric_value(
        metrics,
        "test_log_mae",
        first_float(row, ("test_log_mae", "holdout_log_mae", "test_mae_log", "test_log_error_mae")),
        source,
    )
    update_metric_value(
        metrics,
        "recent_holdout_abs_pct_error_p99",
        first_float(
            row,
            (
                "recent_holdout_abs_pct_error_p99",
                "recent_abs_pct_error_p99",
                "abs_pct_error_p99",
                "recent_holdout_p99",
            ),
        ),
        source,
    )
    update_metric_value(
        metrics,
        "recent_holdout_error_gt_20pct_rate",
        first_float(
            row,
            (
                "recent_holdout_error_gt_20pct_rate",
                "recent_error_gt_20pct_rate",
                "error_gt_20pct_rate",
                "recent_holdout_gt20",
            ),
        ),
        source,
    )

    split = split_value(row)
    log_mae = first_float(row, ("log_mae", "mae_log"))
    p99 = first_float(row, ("abs_pct_error_p99", "p99"))
    gt20 = first_float(row, ("error_gt_20pct_rate", "gt20_rate"))
    if "valid" in split:
        update_metric_value(metrics, "valid_log_mae", log_mae, source)
    if "test" in split:
        update_metric_value(metrics, "test_log_mae", log_mae, source)
    if "recent_holdout" in split or split == "recent":
        update_metric_value(metrics, "recent_holdout_abs_pct_error_p99", p99, source)
        update_metric_value(metrics, "recent_holdout_error_gt_20pct_rate", gt20, source)

    metric_name, metric_value = metric_name_value(row)
    if metric_name and metric_value is not None:
        if "valid" in split and metric_name == "log_mae":
            update_metric_value(metrics, "valid_log_mae", metric_value, source)
        elif "test" in split and metric_name == "log_mae":
            update_metric_value(metrics, "test_log_mae", metric_value, source)
        elif ("recent_holdout" in split or split == "recent") and metric_name in {"abs_pct_error_p99", "p99"}:
            update_metric_value(metrics, "recent_holdout_abs_pct_error_p99", metric_value, source)
        elif ("recent_holdout" in split or split == "recent") and metric_name in {
            "error_gt_20pct_rate",
            "gt20_rate",
        }:
            update_metric_value(metrics, "recent_holdout_error_gt_20pct_rate", metric_value, source)


def has_metric_header(fieldnames: Iterable[str] | None, path: Path) -> bool:
    if not fieldnames:
        return False
    fields = {normalize_key(field) for field in fieldnames if field}
    candidate_fields = {normalize_key(field) for field in CSV_CANDIDATE_FIELDS}
    has_candidate = bool(fields & candidate_fields) or candidate_from_text(path.as_posix()) is not None
    if not has_candidate:
        return False
    direct_metrics = {
        "valid_log_mae",
        "validation_log_mae",
        "valid_mae_log",
        "valid_log_error_mae",
        "test_log_mae",
        "holdout_log_mae",
        "test_mae_log",
        "test_log_error_mae",
        "recent_holdout_abs_pct_error_p99",
        "recent_abs_pct_error_p99",
        "recent_holdout_p99",
        "recent_holdout_error_gt_20pct_rate",
        "recent_error_gt_20pct_rate",
        "recent_holdout_gt20",
    }
    long_metrics = {
        "log_mae",
        "mae_log",
        "abs_pct_error_p99",
        "p99",
        "error_gt_20pct_rate",
        "gt20_rate",
    }
    if fields & direct_metrics:
        return True
    if (fields & {normalize_key(field) for field in SPLIT_FIELDS}) and (fields & long_metrics):
        return True
    if (fields & {normalize_key(field) for field in METRIC_NAME_FIELDS}) and (
        fields & {normalize_key(field) for field in VALUE_FIELDS}
    ):
        return True
    return False


def collect_csv_metrics(epoch_dir: Path, results: dict[str, CandidateMetrics]) -> None:
    for path in sorted(epoch_dir.rglob("*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if not has_metric_header(reader.fieldnames, path):
                    continue
                for raw_row in reader:
                    row = {normalize_key(key): value for key, value in raw_row.items() if key is not None}
                    candidate = candidate_from_row(row, path)
                    if not candidate:
                        continue
                    update_metrics_from_row(results[candidate], row, path.relative_to(epoch_dir))
        except UnicodeDecodeError:
            continue


def flatten_dict(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = normalize_key(str(key))
        next_key = f"{prefix}_{normalized_key}" if prefix else normalized_key
        if isinstance(item, dict):
            flattened.update(flatten_dict(item, next_key))
        else:
            flattened[next_key] = item
    return flattened


def collect_json_metrics(epoch_dir: Path, results: dict[str, CandidateMetrics]) -> None:
    for path in sorted(epoch_dir.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        rows = value if isinstance(value, list) else [value]
        for item in rows:
            if not isinstance(item, dict):
                continue
            flattened = {key: str(raw) for key, raw in flatten_dict(item).items() if raw is not None}
            candidate = candidate_from_row(flattened, path)
            if not candidate:
                continue
            update_metrics_from_row(results[candidate], flattened, path.relative_to(epoch_dir))


def detect_quality_status(epoch_dir: Path) -> str:
    statuses: list[str] = []
    for path in sorted(epoch_dir.rglob("*")):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if "quality" not in lower_name or "report" not in lower_name:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        normalized = text.lower()
        matched_status = False
        for line in normalized.splitlines():
            if "품질 등급" in line or "quality grade" in line:
                if "pass" in line:
                    statuses.append("Pass")
                    matched_status = True
                elif "fail" in line:
                    statuses.append("Fail")
                    matched_status = True
        if not matched_status:
            if "status: pass" in normalized or "quality_status: pass" in normalized or "\npass\n" in normalized:
                statuses.append("Pass")
            elif "status: fail" in normalized or "quality_status: fail" in normalized or "\nfail\n" in normalized:
                statuses.append("Fail")
    if "Fail" in statuses:
        return "Fail"
    if "Pass" in statuses:
        return "Pass"
    return "Unknown"


def collect_epoch_metrics(
    epoch: int,
    epoch_dir: Path,
    expected: Iterable[str] = RUN_CANDIDATES,
) -> dict[str, CandidateMetrics]:
    expected_tuple = tuple(expected)
    results = {candidate: CandidateMetrics(epoch=epoch, candidate=candidate) for candidate in expected_tuple}
    collect_csv_metrics(epoch_dir, results)
    collect_json_metrics(epoch_dir, results)
    quality_status = detect_quality_status(epoch_dir)
    for metrics in results.values():
        metrics.residual_quality_status = quality_status
    missing = [candidate for candidate, metrics in results.items() if not metrics.has_required_metrics()]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"missing required metrics for epoch {epoch}: {missing_text}")
    return results


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def guardrail_for(metrics: CandidateMetrics, baseline: CandidateMetrics) -> tuple[str, tuple[str, ...]]:
    if metrics.candidate == BASELINE_CANDIDATE:
        return "baseline", ()
    if metrics.candidate == SUPPORT_POLICY_CANDIDATE:
        return "support_policy_only", ()
    reasons: list[str] = []
    test_delta = delta(metrics.test_log_mae, baseline.test_log_mae)
    p99_delta = delta(metrics.recent_holdout_abs_pct_error_p99, baseline.recent_holdout_abs_pct_error_p99)
    gt20_delta = delta(metrics.recent_holdout_error_gt_20pct_rate, baseline.recent_holdout_error_gt_20pct_rate)
    if test_delta is not None and test_delta > 0.0005:
        reasons.append(f"test log_mae delta {test_delta:.6f} > 0.0005")
    if p99_delta is not None and p99_delta >= 0.01:
        reasons.append(f"recent_holdout p99 delta {p99_delta:.6f} >= 0.01")
    if gt20_delta is not None and gt20_delta > 0.002:
        reasons.append(f"recent_holdout gt20 delta {gt20_delta:.6f} > 0.002")
    return ("fail" if reasons else "pass", tuple(reasons))


def build_selection_rows(metrics_by_epoch: dict[int, dict[str, CandidateMetrics]]) -> list[SelectionRow]:
    rows: list[SelectionRow] = []
    for epoch in sorted(metrics_by_epoch):
        epoch_metrics = metrics_by_epoch[epoch]
        baseline = epoch_metrics[BASELINE_CANDIDATE]
        for candidate in RUN_CANDIDATES:
            if candidate not in epoch_metrics:
                continue
            metrics = epoch_metrics[candidate]
            guardrail_status, guardrail_reasons = guardrail_for(metrics, baseline)
            if candidate in PRICE_CANDIDATES:
                ranking_role = "ranking_candidate"
            elif candidate == BASELINE_CANDIDATE:
                ranking_role = "baseline"
            else:
                ranking_role = "support_policy"
            rows.append(
                SelectionRow(
                    metrics=metrics,
                    delta_valid_log_mae=delta(metrics.valid_log_mae, baseline.valid_log_mae),
                    delta_test_log_mae=delta(metrics.test_log_mae, baseline.test_log_mae),
                    delta_recent_holdout_abs_pct_error_p99=delta(
                        metrics.recent_holdout_abs_pct_error_p99,
                        baseline.recent_holdout_abs_pct_error_p99,
                    ),
                    delta_recent_holdout_error_gt_20pct_rate=delta(
                        metrics.recent_holdout_error_gt_20pct_rate,
                        baseline.recent_holdout_error_gt_20pct_rate,
                    ),
                    guardrail_status=guardrail_status,
                    guardrail_reasons=guardrail_reasons,
                    ranking_role=ranking_role,
                )
            )
    return rows


def choose_best(rows: Iterable[SelectionRow]) -> SelectionRow | None:
    eligible = [
        row
        for row in rows
        if row.ranking_role == "ranking_candidate"
        and row.guardrail_status == "pass"
        and row.metrics.valid_log_mae is not None
        and row.metrics.test_log_mae is not None
        and row.metrics.recent_holdout_abs_pct_error_p99 is not None
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            row.metrics.valid_log_mae,
            row.metrics.test_log_mae,
            row.metrics.recent_holdout_abs_pct_error_p99,
            SIMPLE_PRIORITY[row.metrics.candidate],
        ),
    )


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_metrics_csv(rows: list[SelectionRow], output_dir: Path) -> None:
    fieldnames = (
        "epoch",
        "candidate",
        "ranking_role",
        "valid_log_mae",
        "test_log_mae",
        "recent_holdout_abs_pct_error_p99",
        "recent_holdout_error_gt_20pct_rate",
        "delta_valid_log_mae_vs_f18",
        "delta_test_log_mae_vs_f18",
        "delta_recent_holdout_abs_pct_error_p99_vs_f18",
        "delta_recent_holdout_error_gt_20pct_rate_vs_f18",
        "guardrail_status",
        "guardrail_reasons",
        "residual_quality_status",
        "source",
    )
    path = output_dir / METRICS_CSV
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.metrics
            writer.writerow(
                {
                    "epoch": metrics.epoch,
                    "candidate": metrics.candidate,
                    "ranking_role": row.ranking_role,
                    "valid_log_mae": fmt_float(metrics.valid_log_mae),
                    "test_log_mae": fmt_float(metrics.test_log_mae),
                    "recent_holdout_abs_pct_error_p99": fmt_float(metrics.recent_holdout_abs_pct_error_p99),
                    "recent_holdout_error_gt_20pct_rate": fmt_float(metrics.recent_holdout_error_gt_20pct_rate),
                    "delta_valid_log_mae_vs_f18": fmt_float(row.delta_valid_log_mae),
                    "delta_test_log_mae_vs_f18": fmt_float(row.delta_test_log_mae),
                    "delta_recent_holdout_abs_pct_error_p99_vs_f18": fmt_float(
                        row.delta_recent_holdout_abs_pct_error_p99
                    ),
                    "delta_recent_holdout_error_gt_20pct_rate_vs_f18": fmt_float(
                        row.delta_recent_holdout_error_gt_20pct_rate
                    ),
                    "guardrail_status": row.guardrail_status,
                    "guardrail_reasons": "; ".join(row.guardrail_reasons),
                    "residual_quality_status": metrics.residual_quality_status,
                    "source": metrics.source,
                }
            )


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    separator = ["---"] * len(header)
    body = rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def write_summary(rows: list[SelectionRow], output_dir: Path, best: SelectionRow | None) -> None:
    table = [
        [
            "epoch",
            "candidate",
            "role",
            "valid log_mae",
            "test log_mae",
            "p99",
            "gt20",
            "guardrail",
            "quality",
        ]
    ]
    for row in rows:
        metrics = row.metrics
        table.append(
            [
                str(metrics.epoch),
                metrics.candidate,
                row.ranking_role,
                fmt_float(metrics.valid_log_mae),
                fmt_float(metrics.test_log_mae),
                fmt_float(metrics.recent_holdout_abs_pct_error_p99),
                fmt_float(metrics.recent_holdout_error_gt_20pct_rate),
                row.guardrail_status,
                metrics.residual_quality_status,
            ]
        )
    epochs = sorted({row.metrics.epoch for row in rows})
    ranking_targets = [candidate for candidate in PRICE_CANDIDATES if any(row.metrics.candidate == candidate for row in rows)]
    has_support_policy = any(row.metrics.candidate == SUPPORT_POLICY_CANDIDATE for row in rows)
    lines = [
        "# E11 Epoch Sweep Summary",
        "",
        "## 요약",
        "",
        "- epoch grid: " + ", ".join(str(epoch) for epoch in epochs),
        "- ranking 대상: " + ", ".join(ranking_targets),
        f"- baseline: {BASELINE_CANDIDATE}",
        f"- support policy: {SUPPORT_POLICY_CANDIDATE if has_support_policy else 'not run'}",
        "- primary score: valid log_mae",
        "- guardrail: test log_mae, recent_holdout p99, recent_holdout gt20",
        "",
    ]
    if best:
        lines.extend(
            [
                "## 최종 후보",
                "",
                f"- best epoch: {best.metrics.epoch}",
                f"- best candidate: {best.metrics.candidate}",
                f"- valid log_mae: {fmt_float(best.metrics.valid_log_mae)}",
                f"- F18 대비 test log_mae delta: {fmt_float(best.delta_test_log_mae)}",
                f"- F18 대비 recent_holdout p99 delta: {fmt_float(best.delta_recent_holdout_abs_pct_error_p99)}",
                f"- F18 대비 recent_holdout gt20 delta: {fmt_float(best.delta_recent_holdout_error_gt_20pct_rate)}",
                f"- residual quality report: {best.metrics.residual_quality_status}",
                "",
            ]
        )
    lines.extend(["## 전체 metrics", "", markdown_table(table), ""])
    (output_dir / SUMMARY_MD).write_text("\n".join(lines), encoding="utf-8")


def write_final_decision(rows: list[SelectionRow], output_dir: Path, best: SelectionRow | None) -> None:
    lines = ["# E11 Epoch Sweep Final Decision", ""]
    if best is None:
        lines.extend(
            [
                "## 결정",
                "",
                "- 채택 후보: 없음",
                "- 이유: guardrail을 통과한 가격 후보가 없습니다.",
                "",
            ]
        )
    else:
        metrics = best.metrics
        lines.extend(
            [
                "## 결정",
                "",
                f"- best epoch: {metrics.epoch}",
                f"- best candidate: {metrics.candidate}",
                f"- primary valid log_mae: {fmt_float(metrics.valid_log_mae)}",
                f"- guardrail: {best.guardrail_status}",
                "",
                "## F18 대비 delta",
                "",
                f"- valid log_mae: {fmt_float(best.delta_valid_log_mae)}",
                f"- test log_mae: {fmt_float(best.delta_test_log_mae)}",
                f"- recent_holdout abs_pct_error_p99: {fmt_float(best.delta_recent_holdout_abs_pct_error_p99)}",
                f"- recent_holdout error_gt_20pct_rate: {fmt_float(best.delta_recent_holdout_error_gt_20pct_rate)}",
                f"- residual quality report: {metrics.residual_quality_status}",
                "",
            ]
        )
    lines.extend(["## guardrail 결과", ""])
    guardrail_table = [["epoch", "candidate", "guardrail", "reasons"]]
    for row in rows:
        if row.ranking_role != "ranking_candidate":
            continue
        guardrail_table.append(
            [
                str(row.metrics.epoch),
                row.metrics.candidate,
                row.guardrail_status,
                "; ".join(row.guardrail_reasons) or "none",
            ]
        )
    lines.extend([markdown_table(guardrail_table), ""])
    lines.extend(["## 보조 정책", ""])
    if any(row.metrics.candidate == SUPPORT_POLICY_CANDIDATE for row in rows):
        lines.append(
            f"- {SUPPORT_POLICY_CANDIDATE}는 가격 후보 ranking에서 제외하고 confidence 보조 정책으로 유지합니다."
        )
    else:
        lines.append(f"- {SUPPORT_POLICY_CANDIDATE}는 이번 필터 실행에서 제외했습니다.")
    lines.append("")
    (output_dir / FINAL_DECISION_MD).write_text("\n".join(lines), encoding="utf-8")


def aggregate_and_write(config: SweepConfig) -> None:
    candidates = expected_candidates(config)
    metrics_by_epoch: dict[int, dict[str, CandidateMetrics]] = {}
    for epoch in config.epochs:
        epoch_dir = config.sweep_output_dir / f"epoch_{epoch}"
        metrics_by_epoch[epoch] = collect_epoch_metrics(epoch, epoch_dir, candidates)
    rows = build_selection_rows(metrics_by_epoch)
    best = choose_best(rows)
    config.sweep_output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(rows, config.sweep_output_dir)
    write_summary(rows, config.sweep_output_dir, best)
    write_final_decision(rows, config.sweep_output_dir, best)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E11 epoch sweep orchestration.")
    parser.add_argument("--summary-only", action="store_true", help="skip runner execution and rebuild sweep reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SweepConfig.from_env()
    config.sweep_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for epoch in config.epochs:
            run_epoch(config, epoch)
    aggregate_and_write(config)
    print(f"wrote {config.sweep_output_dir / METRICS_CSV}", flush=True)
    print(f"wrote {config.sweep_output_dir / SUMMARY_MD}", flush=True)
    print(f"wrote {config.sweep_output_dir / FINAL_DECISION_MD}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
