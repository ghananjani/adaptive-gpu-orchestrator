import math

import pytest

from adaptive_gpu_orchestrator.modeling import (
    PerformanceModelReport,
    PerformanceModelStore,
    fit_metric_model,
    recommend_experiments,
)
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig


def synthetic_throughput(config):
    return (
        100.0
        + 20.0 * math.log2(config.tensor_parallel)
        + 3.0 * math.log2(config.data_parallel)
        + 8.0 * math.log2(config.batch_size)
        + 4.0 * math.log2(config.concurrency)
    )


def test_log_linear_ridge_predicts_known_scaling_surface(tmp_path):
    configs = [
        ParallelismConfig(),
        ParallelismConfig(tensor_parallel=2),
        ParallelismConfig(tensor_parallel=4),
        ParallelismConfig(batch_size=2),
        ParallelismConfig(concurrency=2),
        ParallelismConfig(data_parallel=2),
        ParallelismConfig(
            tensor_parallel=2,
            data_parallel=2,
            batch_size=2,
            concurrency=2,
        ),
        ParallelismConfig(
            tensor_parallel=4,
            data_parallel=2,
            batch_size=4,
            concurrency=2,
        ),
    ]
    targets = [synthetic_throughput(config) for config in configs]

    metric = fit_metric_model(
        metric="throughput_per_s",
        configs=configs,
        targets=targets,
        ridge_lambda=1e-6,
    )

    unseen = ParallelismConfig(
        tensor_parallel=8,
        data_parallel=2,
        batch_size=4,
        concurrency=2,
    )
    assert metric.sample_count == len(configs)
    assert metric.distinct_config_count == len(configs)
    assert metric.cv_rmse is not None
    assert metric.predict(unseen) == pytest.approx(
        synthetic_throughput(unseen),
        abs=0.05,
    )

    report = PerformanceModelReport(
        campaign_ids=("campaign-a",),
        metrics={"throughput_per_s": metric},
        observed_configs=tuple(configs),
        hardware_signatures=("gpu_models=H100",),
        ridge_lambda=1e-6,
    )

    recommendation = recommend_experiments(
        report,
        [
            ParallelismConfig(tensor_parallel=8),
            unseen,
        ],
        metric="throughput_per_s",
        exploration_weight=0.0,
        top_k=1,
    )

    assert recommendation.direction == "maximize"
    assert recommendation.recommendations[0].config == unseen

    store = PerformanceModelStore(tmp_path / "models")
    store.save(report)
    loaded = store.load(report.id)
    assert loaded.to_dict() == report.to_dict()


def test_custom_metric_requires_explicit_direction():
    configs = [
        ParallelismConfig(),
        ParallelismConfig(tensor_parallel=2),
        ParallelismConfig(tensor_parallel=4),
    ]
    metric = fit_metric_model(
        metric="extra_custom",
        configs=configs,
        targets=[1.0, 2.0, 3.0],
        ridge_lambda=0.1,
    )
    report = PerformanceModelReport(
        campaign_ids=("campaign-a",),
        metrics={"extra_custom": metric},
        observed_configs=tuple(configs),
        hardware_signatures=("sig",),
        ridge_lambda=0.1,
    )

    with pytest.raises(ValueError, match="direction is required"):
        recommend_experiments(
            report,
            [ParallelismConfig(tensor_parallel=8)],
            metric="extra_custom",
        )
