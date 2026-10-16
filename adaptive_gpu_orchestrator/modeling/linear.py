from __future__ import annotations

import math

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

from .models import (
    MetricPerformanceModel,
    config_feature_vector,
)


def _solve_linear_system(
    matrix: list[list[float]],
    vector: list[float],
) -> list[float]:
    size = len(vector)
    augmented = [
        [*matrix[row], vector[row]]
        for row in range(size)
    ]

    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("ridge system is numerically singular")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )

        divisor = augmented[column][column]
        augmented[column] = [
            value / divisor
            for value in augmented[column]
        ]

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                )
            ]

    return [
        augmented[row][-1]
        for row in range(size)
    ]


def _fit_coefficients(
    configs: list[ParallelismConfig],
    targets: list[float],
    ridge_lambda: float,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
]:
    features = [
        config_feature_vector(config)
        for config in configs
    ]
    feature_count = len(features[0])
    sample_count = len(features)

    means = tuple(
        sum(row[column] for row in features) / sample_count
        for column in range(feature_count)
    )
    scales = []
    for column, mean in enumerate(means):
        variance = (
            sum(
                (row[column] - mean) ** 2
                for row in features
            )
            / sample_count
        )
        scale = math.sqrt(variance)
        scales.append(scale if scale > 1e-12 else 1.0)
    scale_tuple = tuple(scales)

    standardized = [
        [
            (value - means[column]) / scale_tuple[column]
            for column, value in enumerate(row)
        ]
        for row in features
    ]

    intercept = sum(targets) / len(targets)
    centered_targets = [
        target - intercept
        for target in targets
    ]

    gram = [
        [0.0 for _ in range(feature_count)]
        for _ in range(feature_count)
    ]
    rhs = [0.0 for _ in range(feature_count)]

    for row, target in zip(standardized, centered_targets):
        for left in range(feature_count):
            rhs[left] += row[left] * target
            for right in range(feature_count):
                gram[left][right] += row[left] * row[right]

    for index in range(feature_count):
        gram[index][index] += ridge_lambda

    coefficients = tuple(
        _solve_linear_system(gram, rhs)
    )
    return coefficients, means, scale_tuple, intercept


def _predict(
    config: ParallelismConfig,
    coefficients: tuple[float, ...],
    means: tuple[float, ...],
    scales: tuple[float, ...],
    intercept: float,
) -> float:
    features = config_feature_vector(config)
    return intercept + sum(
        coefficient * ((value - mean) / scale)
        for coefficient, value, mean, scale in zip(
            coefficients,
            features,
            means,
            scales,
        )
    )


def _errors(
    actual: list[float],
    predicted: list[float],
) -> tuple[float, float]:
    residuals = [
        estimate - target
        for estimate, target in zip(predicted, actual)
    ]
    rmse = math.sqrt(
        sum(value * value for value in residuals)
        / len(residuals)
    )
    mae = sum(abs(value) for value in residuals) / len(residuals)
    return rmse, mae


def fit_metric_model(
    *,
    metric: str,
    configs: list[ParallelismConfig],
    targets: list[float],
    ridge_lambda: float = 1.0,
) -> MetricPerformanceModel:
    if len(configs) != len(targets):
        raise ValueError("configs and targets must have equal length")
    if len(configs) < 2:
        raise ValueError("metric model requires at least two samples")
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")

    coefficients, means, scales, intercept = _fit_coefficients(
        configs,
        targets,
        ridge_lambda,
    )
    predicted = [
        _predict(
            config,
            coefficients,
            means,
            scales,
            intercept,
        )
        for config in configs
    ]
    fit_rmse, fit_mae = _errors(targets, predicted)

    target_mean = sum(targets) / len(targets)
    total_variance = sum(
        (target - target_mean) ** 2
        for target in targets
    )
    residual_variance = sum(
        (estimate - target) ** 2
        for estimate, target in zip(predicted, targets)
    )
    r2 = (
        None
        if total_variance <= 1e-12
        else 1.0 - residual_variance / total_variance
    )

    cv_rmse = None
    cv_mae = None
    if len(configs) >= 4:
        cv_predictions = []
        for holdout in range(len(configs)):
            train_configs = [
                config
                for index, config in enumerate(configs)
                if index != holdout
            ]
            train_targets = [
                target
                for index, target in enumerate(targets)
                if index != holdout
            ]
            (
                fold_coefficients,
                fold_means,
                fold_scales,
                fold_intercept,
            ) = _fit_coefficients(
                train_configs,
                train_targets,
                ridge_lambda,
            )
            cv_predictions.append(
                _predict(
                    configs[holdout],
                    fold_coefficients,
                    fold_means,
                    fold_scales,
                    fold_intercept,
                )
            )
        cv_rmse, cv_mae = _errors(targets, cv_predictions)

    return MetricPerformanceModel(
        metric=metric,
        coefficients=coefficients,
        feature_means=means,
        feature_scales=scales,
        intercept=intercept,
        sample_count=len(configs),
        distinct_config_count=len(set(configs)),
        fit_rmse=fit_rmse,
        fit_mae=fit_mae,
        r2=r2,
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
    )
