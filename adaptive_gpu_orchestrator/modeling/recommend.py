from __future__ import annotations

import math
from collections.abc import Iterable

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

from .models import (
    ExperimentRecommendation,
    PerformanceModelReport,
    RecommendationReport,
    config_feature_vector,
)

_METRIC_DIRECTIONS = {
    "throughput_per_s": "maximize",
    "gpu_util_pct": "maximize",
    "p99_latency_ms": "minimize",
    "energy_joules": "minimize",
    "memory_pressure_pct": "minimize",
    "duration_s": "minimize",
}


def infer_metric_direction(metric: str) -> str:
    try:
        return _METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise ValueError(
            f"direction is required for custom metric: {metric}"
        ) from exc


def _distance(
    left: ParallelismConfig,
    right: ParallelismConfig,
    *,
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> float:
    left_values = config_feature_vector(left)
    right_values = config_feature_vector(right)
    squared = 0.0
    for left_value, right_value, mean, scale in zip(
        left_values,
        right_values,
        means,
        scales,
    ):
        normalized_left = (left_value - mean) / scale
        normalized_right = (right_value - mean) / scale
        squared += (normalized_left - normalized_right) ** 2
    return math.sqrt(squared / len(left_values))


def recommend_experiments(
    model: PerformanceModelReport,
    candidates: Iterable[ParallelismConfig],
    *,
    metric: str = "throughput_per_s",
    direction: str | None = None,
    top_k: int = 5,
    exploration_weight: float = 0.25,
    include_observed: bool = False,
) -> RecommendationReport:
    if metric not in model.metrics:
        raise ValueError(f"metric is not available in model: {metric}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= exploration_weight <= 1.0:
        raise ValueError("exploration_weight must be between 0 and 1")

    resolved_direction = direction or infer_metric_direction(metric)
    if resolved_direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")

    metric_model = model.metrics[metric]
    observed = set(model.observed_configs)
    unique_candidates = sorted(
        set(candidates),
        key=lambda config: config.label(),
    )
    eligible = [
        config
        for config in unique_candidates
        if include_observed or config not in observed
    ]
    if not eligible:
        raise ValueError("no eligible candidate configurations remain")

    predictions = {
        config: metric_model.predict(config)
        for config in eligible
    }
    values = list(predictions.values())
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum

    exploration = {}
    for config in eligible:
        if not observed:
            exploration[config] = 0.0
            continue
        exploration[config] = min(
            _distance(
                config,
                existing,
                means=metric_model.feature_means,
                scales=metric_model.feature_scales,
            )
            for existing in observed
        )

    max_distance = max(exploration.values(), default=0.0)
    recommendations = []
    for config in eligible:
        predicted = predictions[config]
        if span <= 1e-12:
            exploitation_score = 0.5
        elif resolved_direction == "maximize":
            exploitation_score = (predicted - minimum) / span
        else:
            exploitation_score = (maximum - predicted) / span

        exploration_score = (
            0.0
            if max_distance <= 1e-12
            else exploration[config] / max_distance
        )
        acquisition_score = (
            (1.0 - exploration_weight) * exploitation_score
            + exploration_weight * exploration_score
        )
        recommendations.append(
            ExperimentRecommendation(
                config=config,
                predicted_value=predicted,
                acquisition_score=acquisition_score,
                exploitation_score=exploitation_score,
                exploration_score=exploration_score,
                observed=config in observed,
            )
        )

    recommendations.sort(
        key=lambda item: (
            -item.acquisition_score,
            -item.exploitation_score,
            item.config.label(),
        )
    )
    selected = recommendations[:top_k]

    findings = [
        (
            "acquisition score blends model-predicted objective quality with "
            "distance from previously measured configurations"
        )
    ]
    if any(item.exploration_score > 0.8 for item in selected):
        findings.append(
            "one or more recommendations are far from measured configurations; "
            "treat predicted values as extrapolations"
        )

    return RecommendationReport(
        model_id=model.id,
        metric=metric,
        direction=resolved_direction,
        candidate_count=len(eligible),
        recommendations=selected,
        exploration_weight=exploration_weight,
        findings=findings,
    )
