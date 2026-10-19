"""Paired hardware-held-out evaluation of two experimental ridge predictors.

A held-out platform is necessarily unseen by the conditioned estimator.
Extrapolation is *only* enabled here for offline error measurement, never
for scheduling or production recommendations.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

from .conditioned import HardwareTrainingSample, fit_hardware_conditioned_model
from .hardware import HardwareFeatures
from .linear import fit_metric_model


@dataclass(frozen=True, slots=True)
class LabeledHardwareTrial:
    platform: str
    config: ParallelismConfig
    hardware: HardwareFeatures
    target: float

    def __post_init__(self) -> None:
        if not self.platform.strip():
            raise ValueError("platform must not be empty")
        if isinstance(self.target, bool) or not isinstance(self.target, (int, float)):
            raise TypeError("target must be numeric")
        if not math.isfinite(self.target):
            raise ValueError("target must be finite")


@dataclass(frozen=True, slots=True)
class PairedHardwareFold:
    held_out_platform: str
    train_samples: int
    test_samples: int
    config_rmse: float
    conditioned_rmse: float
    config_mae: float
    conditioned_mae: float
    rmse_reduction_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_platform": self.held_out_platform,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "config_rmse": self.config_rmse,
            "conditioned_rmse": self.conditioned_rmse,
            "config_mae": self.config_mae,
            "conditioned_mae": self.conditioned_mae,
            "rmse_reduction_pct": self.rmse_reduction_pct,
        }


@dataclass(frozen=True, slots=True)
class PairedHardwareEvaluation:
    metric: str
    folds: tuple[PairedHardwareFold, ...]
    config_pooled_rmse: float
    conditioned_pooled_rmse: float
    config_pooled_mae: float
    conditioned_pooled_mae: float
    rmse_reduction_pct: float | None
    test_samples: int
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "folds": [fold.to_dict() for fold in self.folds],
            "config_pooled_rmse": self.config_pooled_rmse,
            "conditioned_pooled_rmse": self.conditioned_pooled_rmse,
            "config_pooled_mae": self.config_pooled_mae,
            "conditioned_pooled_mae": self.conditioned_pooled_mae,
            "rmse_reduction_pct": self.rmse_reduction_pct,
            "test_samples": self.test_samples,
            "findings": list(self.findings),
        }


def _rmse(errors: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in errors) / len(errors))


def _mae(errors: list[float]) -> float:
    return math.fsum(abs(value) for value in errors) / len(errors)


def _reduction(baseline: float, candidate: float) -> float | None:
    return None if baseline == 0.0 else 100.0 * (baseline - candidate) / baseline


def compare_hardware_heldout_models(
    trials: Iterable[LabeledHardwareTrial],
    *,
    metric: str = "throughput_per_s",
    ridge_lambda: float = 1.0,
    min_train_samples: int = 4,
) -> PairedHardwareEvaluation:
    """Measure both models on identical platform-held-out observations.

    At least three distinct hardware descriptors/platforms are required, since
    the conditioned model itself needs >=2 hardware descriptors in every fold.
    Mixed descriptors for the same platform are rejected to avoid an ambiguous
    holdout. No saved model or GPU resource is mutated.
    """
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    if min_train_samples < 4:
        raise ValueError("min_train_samples must be at least four")
    grouped: dict[str, list[LabeledHardwareTrial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.platform].append(trial)
    if len(grouped) < 3:
        raise ValueError("at least three platforms are required for conditioned holdout")
    for platform, values in grouped.items():
        if len({trial.hardware for trial in values}) != 1:
            raise ValueError(f"platform has multiple hardware descriptors: {platform}")
    if len({values[0].hardware for values in grouped.values()}) < 3:
        raise ValueError("at least three distinct hardware descriptors are required")
    for values in grouped.values():
        values.sort(key=lambda t: (t.config.label(), t.target))

    folds = []
    config_errors_all: list[float] = []
    conditioned_errors_all: list[float] = []
    for held_out in sorted(grouped):
        train = [trial for platform in sorted(grouped) if platform != held_out
                 for trial in grouped[platform]]
        test = grouped[held_out]
        if len(train) < min_train_samples or len({t.config for t in train}) < 2:
            raise ValueError(f"insufficient training diversity for {held_out}")
        config_model = fit_metric_model(
            metric=metric,
            configs=[t.config for t in train],
            targets=[float(t.target) for t in train],
            ridge_lambda=ridge_lambda,
        )
        conditioned_model = fit_hardware_conditioned_model(
            [HardwareTrainingSample(t.config, t.hardware, t.target) for t in train],
            metric=metric,
            ridge_lambda=ridge_lambda,
        )
        config_errors = [config_model.predict(t.config) - t.target for t in test]
        # The target platform is explicitly unseen. This bypass is limited to
        # quantitative holdout evaluation; it is never a deployment approval.
        conditioned_errors = [
            conditioned_model.predict(t.config, t.hardware, allow_ood=True) - t.target
            for t in test
        ]
        config_errors_all.extend(config_errors)
        conditioned_errors_all.extend(conditioned_errors)
        config_rmse = _rmse(config_errors)
        conditioned_rmse = _rmse(conditioned_errors)
        folds.append(PairedHardwareFold(
            held_out_platform=held_out,
            train_samples=len(train),
            test_samples=len(test),
            config_rmse=config_rmse,
            conditioned_rmse=conditioned_rmse,
            config_mae=_mae(config_errors),
            conditioned_mae=_mae(conditioned_errors),
            rmse_reduction_pct=_reduction(config_rmse, conditioned_rmse),
        ))
    config_pooled = _rmse(config_errors_all)
    conditioned_pooled = _rmse(conditioned_errors_all)
    return PairedHardwareEvaluation(
        metric=metric,
        folds=tuple(folds),
        config_pooled_rmse=config_pooled,
        conditioned_pooled_rmse=conditioned_pooled,
        config_pooled_mae=_mae(config_errors_all),
        conditioned_pooled_mae=_mae(conditioned_errors_all),
        rmse_reduction_pct=_reduction(config_pooled, conditioned_pooled),
        test_samples=len(config_errors_all),
        findings=(
            "Both estimators are trained only on non-held-out platforms.",
            "Held-out conditioned predictions explicitly bypass OOD only for offline evaluation.",
            "Error reduction may be negative; no superiority or calibrated uncertainty is assumed.",
            "Require matched workload, precision, and measurement definitions across platforms.",
        ),
    )
