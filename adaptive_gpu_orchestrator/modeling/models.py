from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

MODEL_SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "log2_tensor_parallel",
    "log2_pipeline_parallel",
    "log2_data_parallel",
    "log2_batch_size",
    "log2_concurrency",
)


def config_feature_vector(config: ParallelismConfig) -> tuple[float, ...]:
    return (
        math.log2(config.tensor_parallel),
        math.log2(config.pipeline_parallel),
        math.log2(config.data_parallel),
        math.log2(config.batch_size),
        math.log2(config.concurrency),
    )


@dataclass(frozen=True, slots=True)
class MetricPerformanceModel:
    metric: str
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    sample_count: int
    distinct_config_count: int
    fit_rmse: float
    fit_mae: float
    r2: float | None = None
    cv_rmse: float | None = None
    cv_mae: float | None = None

    def __post_init__(self) -> None:
        expected = len(FEATURE_NAMES)
        for values in (
            self.coefficients,
            self.feature_means,
            self.feature_scales,
        ):
            if len(values) != expected:
                raise ValueError(
                    f"metric model requires {expected} feature values"
                )

    def predict(self, config: ParallelismConfig) -> float:
        features = config_feature_vector(config)
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(
                features,
                self.feature_means,
                self.feature_scales,
            )
        ]
        return self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(
                self.coefficients,
                standardized,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "feature_names": list(FEATURE_NAMES),
            "coefficients": list(self.coefficients),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "intercept": self.intercept,
            "sample_count": self.sample_count,
            "distinct_config_count": self.distinct_config_count,
            "fit_rmse": self.fit_rmse,
            "fit_mae": self.fit_mae,
            "r2": self.r2,
            "cv_rmse": self.cv_rmse,
            "cv_mae": self.cv_mae,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricPerformanceModel:
        feature_names = tuple(data.get("feature_names", FEATURE_NAMES))
        if feature_names != FEATURE_NAMES:
            raise ValueError("unsupported performance-model feature schema")
        return cls(
            metric=str(data["metric"]),
            coefficients=tuple(float(value) for value in data["coefficients"]),
            feature_means=tuple(float(value) for value in data["feature_means"]),
            feature_scales=tuple(float(value) for value in data["feature_scales"]),
            intercept=float(data["intercept"]),
            sample_count=int(data["sample_count"]),
            distinct_config_count=int(data["distinct_config_count"]),
            fit_rmse=float(data["fit_rmse"]),
            fit_mae=float(data["fit_mae"]),
            r2=None if data.get("r2") is None else float(data["r2"]),
            cv_rmse=(
                None if data.get("cv_rmse") is None else float(data["cv_rmse"])
            ),
            cv_mae=(
                None if data.get("cv_mae") is None else float(data["cv_mae"])
            ),
        )


@dataclass(slots=True)
class PerformanceModelReport:
    campaign_ids: tuple[str, ...]
    metrics: dict[str, MetricPerformanceModel]
    observed_configs: tuple[ParallelismConfig, ...]
    hardware_signatures: tuple[str, ...]
    ridge_lambda: float
    findings: list[str] = field(default_factory=list)
    schema_version: int = MODEL_SCHEMA_VERSION
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "campaign_ids": list(self.campaign_ids),
            "metrics": {
                name: model.to_dict()
                for name, model in sorted(self.metrics.items())
            },
            "observed_configs": [
                config.to_dict()
                for config in self.observed_configs
            ],
            "hardware_signatures": list(self.hardware_signatures),
            "ridge_lambda": self.ridge_lambda,
            "findings": list(self.findings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceModelReport:
        schema_version = int(data.get("schema_version", 0))
        if schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported performance-model schema version: {schema_version}"
            )
        return cls(
            id=str(data["id"]),
            schema_version=schema_version,
            campaign_ids=tuple(
                str(value)
                for value in data.get("campaign_ids", [])
            ),
            metrics={
                str(name): MetricPerformanceModel.from_dict(payload)
                for name, payload in data.get("metrics", {}).items()
            },
            observed_configs=tuple(
                ParallelismConfig.from_dict(item)
                for item in data.get("observed_configs", [])
            ),
            hardware_signatures=tuple(
                str(value)
                for value in data.get("hardware_signatures", [])
            ),
            ridge_lambda=float(data["ridge_lambda"]),
            findings=[
                str(value)
                for value in data.get("findings", [])
            ],
        )


@dataclass(frozen=True, slots=True)
class ExperimentRecommendation:
    config: ParallelismConfig
    predicted_value: float
    acquisition_score: float
    exploitation_score: float
    exploration_score: float
    observed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "config_label": self.config.label(),
            "predicted_value": self.predicted_value,
            "acquisition_score": self.acquisition_score,
            "exploitation_score": self.exploitation_score,
            "exploration_score": self.exploration_score,
            "observed": self.observed,
        }


@dataclass(slots=True)
class RecommendationReport:
    model_id: str
    metric: str
    direction: str
    candidate_count: int
    recommendations: list[ExperimentRecommendation]
    exploration_weight: float
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "metric": self.metric,
            "direction": self.direction,
            "candidate_count": self.candidate_count,
            "exploration_weight": self.exploration_weight,
            "recommendations": [
                item.to_dict()
                for item in self.recommendations
            ],
            "findings": list(self.findings),
        }
