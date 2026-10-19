"""Experimental hardware-conditioned ridge regression (no deployment actions)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

from .hardware import HARDWARE_FEATURE_NAMES, HardwareFeatures, assess_hardware_ood
from .linear import _solve_linear_system
from .models import FEATURE_NAMES, config_feature_vector

CONDITIONED_FEATURE_NAMES = FEATURE_NAMES + HARDWARE_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class HardwareTrainingSample:
    config: ParallelismConfig
    hardware: HardwareFeatures
    target: float

    def __post_init__(self) -> None:
        if isinstance(self.target, bool) or not isinstance(self.target, (float, int)):
            raise TypeError("target must be numeric")
        if not math.isfinite(self.target):
            raise ValueError("target must be finite")


@dataclass(frozen=True, slots=True)
class ConditionedRidgeModel:
    metric: str
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: float
    training_hardware: tuple[HardwareFeatures, ...]
    sample_count: int
    fit_rmse: float
    ridge_lambda: float

    def predict(
        self,
        config: ParallelismConfig,
        hardware: HardwareFeatures,
        *,
        allow_ood: bool = False,
    ) -> float:
        diagnostic = assess_hardware_ood(self.training_hardware, hardware)
        if diagnostic.out_of_distribution and not allow_ood:
            raise ValueError(
                "hardware outside training distribution; "
                "independent benchmarking is required (or explicitly allow_ood)"
            )
        values = config_feature_vector(config) + hardware.vector()
        return self.intercept + math.fsum(
            coefficient * ((value - mean) / scale)
            for coefficient, value, mean, scale in zip(
                self.coefficients, values, self.means, self.scales
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "hardware-conditioned-ridge",
            "metric": self.metric,
            "feature_names": list(CONDITIONED_FEATURE_NAMES),
            "coefficients": list(self.coefficients),
            "means": list(self.means),
            "scales": list(self.scales),
            "intercept": self.intercept,
            "training_hardware": [item.to_dict() for item in self.training_hardware],
            "sample_count": self.sample_count,
            "fit_rmse": self.fit_rmse,
            "ridge_lambda": self.ridge_lambda,
            "warning": "Not calibrated; out-of-hardware-range prediction is blocked by default.",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionedRidgeModel:
        if data.get("kind") != "hardware-conditioned-ridge" or data.get("schema_version") != 1:
            raise ValueError("unsupported conditioned model schema")
        if tuple(data.get("feature_names", ())) != CONDITIONED_FEATURE_NAMES:
            raise ValueError("conditioned feature schema mismatch")
        model = cls(
            metric=str(data["metric"]),
            coefficients=tuple(float(v) for v in data["coefficients"]),
            means=tuple(float(v) for v in data["means"]),
            scales=tuple(float(v) for v in data["scales"]),
            intercept=float(data["intercept"]),
            training_hardware=tuple(
                HardwareFeatures.from_dict(item) for item in data["training_hardware"]
            ),
            sample_count=int(data["sample_count"]),
            fit_rmse=float(data["fit_rmse"]),
            ridge_lambda=float(data["ridge_lambda"]),
        )
        width = len(CONDITIONED_FEATURE_NAMES)
        if any(len(v) != width for v in (model.coefficients, model.means, model.scales)):
            raise ValueError("conditioned coefficient width mismatch")
        if not model.training_hardware or any(
            not math.isfinite(v) for v in (
                *model.coefficients, *model.means, *model.scales, model.intercept
            )
        ) or any(v <= 0 for v in model.scales):
            raise ValueError("invalid conditioned model parameters")
        return model


def fit_hardware_conditioned_model(
    samples: list[HardwareTrainingSample],
    *,
    metric: str = "throughput_per_s",
    ridge_lambda: float = 1.0,
) -> ConditionedRidgeModel:
    """Fit an additive log-feature model on explicitly measured hardware.

    Hardware must vary across samples; use the v1.6 model for single-platform work.
    All samples must describe comparable workload, precision and benchmark units.
    """
    if len(samples) < 4:
        raise ValueError("at least four samples are required")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    hardware = tuple(sorted({sample.hardware for sample in samples}, key=lambda h: h.vector()))
    if len(hardware) < 2:
        raise ValueError("at least two distinct hardware descriptors are required")
    ordered = sorted(
        samples, key=lambda item: (
            item.hardware.vector(), item.config.label(), float(item.target)
        )
    )
    rows = [config_feature_vector(sample.config) + sample.hardware.vector()
            for sample in ordered]
    targets = [float(sample.target) for sample in ordered]
    n, width = len(rows), len(CONDITIONED_FEATURE_NAMES)
    means = tuple(math.fsum(row[i] for row in rows) / n for i in range(width))
    scales = tuple(max(
        math.sqrt(math.fsum((row[i] - means[i]) ** 2 for row in rows) / n), 1e-12
    ) for i in range(width))
    normalized = [
        [(value - means[i]) / scales[i] for i, value in enumerate(row)] for row in rows
    ]
    intercept = math.fsum(targets) / n
    gram = [
        [math.fsum(row[i] * row[j] for row in normalized) +
         (ridge_lambda if i == j else 0.0) for j in range(width)]
        for i in range(width)
    ]
    rhs = [
        math.fsum(row[i] * (target - intercept)
                  for row, target in zip(normalized, targets))
        for i in range(width)
    ]
    coeffs = tuple(_solve_linear_system(gram, rhs))
    estimates = [
        intercept + math.fsum(c * v for c, v in zip(coeffs, row))
        for row in normalized
    ]
    rmse = math.sqrt(
        math.fsum((pred - target) ** 2 for pred, target in zip(estimates, targets)) / n
    )
    return ConditionedRidgeModel(
        metric=metric,
        coefficients=coeffs,
        means=means,
        scales=scales,
        intercept=intercept,
        training_hardware=hardware,
        sample_count=n,
        fit_rmse=rmse,
        ridge_lambda=ridge_lambda,
    )
