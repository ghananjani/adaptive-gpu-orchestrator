"""Cross-platform evaluation for configuration-only performance models.

This module evaluates transfer *failure or success*; it does not fit a
hardware-aware predictor and must not be represented as calibrated inference.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

from .linear import fit_metric_model


@dataclass(frozen=True, slots=True)
class HardwareHoldout:
    held_out_platform: str
    train_samples: int
    test_samples: int
    train_platforms: tuple[str, ...]
    rmse: float
    mae: float
    mean_baseline_rmse: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_platform": self.held_out_platform,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_platforms": list(self.train_platforms),
            "rmse": self.rmse,
            "mae": self.mae,
            "mean_baseline_rmse": self.mean_baseline_rmse,
        }


@dataclass(frozen=True, slots=True)
class HardwareTransferEvaluation:
    metric: str
    folds: tuple[HardwareHoldout, ...]
    pooled_rmse: float
    pooled_mae: float
    sample_count: int
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "folds": [fold.to_dict() for fold in self.folds],
            "pooled_rmse": self.pooled_rmse,
            "pooled_mae": self.pooled_mae,
            "sample_count": self.sample_count,
            "findings": list(self.findings),
        }


def _config(row: dict[str, Any]) -> ParallelismConfig:
    return ParallelismConfig(
        tensor_parallel=int(row["tensor_parallel"]),
        pipeline_parallel=int(row["pipeline_parallel"]),
        data_parallel=int(row["data_parallel"]),
        batch_size=int(row["batch_size"]),
        concurrency=int(row["concurrency"]),
    )


def evaluate_hardware_transfer(
    rows: Iterable[dict[str, Any]],
    *,
    metric: str = "throughput_per_s",
    platform_field: str = "gpu_models",
    ridge_lambda: float = 1.0,
    min_train_samples: int = 3,
) -> HardwareTransferEvaluation:
    """Evaluate held-out *platforms*, never random trials from the same platform.

    Rows use the existing campaign-trial export schema. Missing platforms and
    nonfinite/missing targets are excluded. Each training fold must have at
    least two distinct configurations; incomplete folds are rejected.
    """
    if ridge_lambda <= 0 or not math.isfinite(ridge_lambda):
        raise ValueError("ridge_lambda must be positive and finite")
    if min_train_samples < 2:
        raise ValueError("min_train_samples must be at least two")
    by_platform: dict[str, list[tuple[ParallelismConfig, float]]] = defaultdict(list)
    for row in rows:
        if int(row.get("exit_code", 0)) != 0:
            continue
        platform = row.get(platform_field)
        value = row.get(metric)
        if platform is None or not str(platform).strip():
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        by_platform[str(platform)].append((_config(row), float(value)))

    if len(by_platform) < 2:
        raise ValueError("cross-platform evaluation requires at least two platforms")

    for samples in by_platform.values():
        samples.sort(key=lambda item: (item[0].label(), item[1]))

    folds = []
    squared_errors = []
    absolute_errors = []
    for held_out in sorted(by_platform):
        train_platforms = tuple(p for p in sorted(by_platform) if p != held_out)
        train = [sample for p in train_platforms for sample in by_platform[p]]
        test = by_platform[held_out]
        if len(train) < min_train_samples or len({config for config, _ in train}) < 2:
            raise ValueError(
                f"insufficient training diversity when holding out {held_out}"
            )
        fitted = fit_metric_model(
            metric=metric,
            configs=[config for config, _ in train],
            targets=[value for _, value in train],
            ridge_lambda=ridge_lambda,
        )
        baseline = sum(value for _, value in train) / len(train)
        errors = [fitted.predict(config) - value for config, value in test]
        baseline_errors = [baseline - value for _, value in test]
        squared_errors.extend(error * error for error in errors)
        absolute_errors.extend(abs(error) for error in errors)
        folds.append(
            HardwareHoldout(
                held_out_platform=held_out,
                train_samples=len(train),
                test_samples=len(test),
                train_platforms=train_platforms,
                rmse=math.sqrt(sum(error * error for error in errors) / len(errors)),
                mae=sum(abs(error) for error in errors) / len(errors),
                mean_baseline_rmse=math.sqrt(
                    sum(error * error for error in baseline_errors) / len(test)
                ),
            )
        )
    return HardwareTransferEvaluation(
        metric=metric,
        folds=tuple(folds),
        pooled_rmse=math.sqrt(sum(squared_errors) / len(squared_errors)),
        pooled_mae=sum(absolute_errors) / len(absolute_errors),
        sample_count=len(absolute_errors),
        findings=(
            "Held-out platform evaluations measure extrapolation, not calibrated uncertainty.",
            "Predictor uses configuration features only; hardware is used solely for splitting.",
            "Platform comparisons may be confounded by workload, software, and configuration coverage.",
            "A mean-target baseline is reported per fold; no transfer-learning claim is made.",
        ),
    )
