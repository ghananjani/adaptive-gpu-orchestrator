from __future__ import annotations

from collections.abc import Iterable

from adaptive_gpu_orchestrator.campaigns import (
    CampaignReport,
    campaign_trial_rows,
)
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationStore,
    ParallelismConfig,
)

from .linear import fit_metric_model
from .models import PerformanceModelReport

DEFAULT_MODEL_METRICS = (
    "throughput_per_s",
    "p99_latency_ms",
    "energy_joules",
    "memory_pressure_pct",
)


def _hardware_signature(row: dict) -> str:
    fields = (
        "gpu_models",
        "gpu_count",
        "driver_version",
        "cuda_runtime_version",
        "cuda_toolkit_version",
        "nccl_version",
        "container_image_digest",
    )
    return "|".join(
        f"{name}={row.get(name) if row.get(name) is not None else ''}"
        for name in fields
    )


def _row_config(row: dict) -> ParallelismConfig:
    return ParallelismConfig(
        tensor_parallel=int(row["tensor_parallel"]),
        pipeline_parallel=int(row["pipeline_parallel"]),
        data_parallel=int(row["data_parallel"]),
        batch_size=int(row["batch_size"]),
        concurrency=int(row["concurrency"]),
    )


def train_performance_model(
    campaigns: Iterable[CampaignReport],
    *,
    optimization_store: OptimizationStore,
    metrics: Iterable[str] | None = None,
    ridge_lambda: float = 1.0,
    min_samples: int = 3,
    allow_mixed_hardware: bool = False,
) -> PerformanceModelReport:
    if min_samples < 2:
        raise ValueError("min_samples must be at least two")
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")

    campaign_list = list(campaigns)
    if not campaign_list:
        raise ValueError("performance modeling requires at least one campaign")

    rows: list[dict] = []
    for campaign in campaign_list:
        rows.extend(
            campaign_trial_rows(
                campaign,
                optimization_store=optimization_store,
            )
        )

    successful_rows = [
        row
        for row in rows
        if int(row.get("exit_code", 0)) == 0
    ]
    if not successful_rows:
        raise ValueError("campaigns contain no successful optimizer trials")

    hardware_signatures = tuple(
        sorted(
            {
                _hardware_signature(row)
                for row in successful_rows
            }
        )
    )
    if len(hardware_signatures) > 1 and not allow_mixed_hardware:
        raise ValueError(
            "campaigns span multiple hardware/software signatures; "
            "train separate models or use the explicit mixed-hardware override"
        )

    requested = tuple(metrics or DEFAULT_MODEL_METRICS)
    explicit_metrics = metrics is not None
    trained = {}
    findings: list[str] = []

    for metric in requested:
        metric_rows = [
            row
            for row in successful_rows
            if isinstance(row.get(metric), (int, float))
        ]
        configs = [
            _row_config(row)
            for row in metric_rows
        ]
        targets = [
            float(row[metric])
            for row in metric_rows
        ]
        distinct = len(set(configs))

        if len(configs) < min_samples or distinct < 2:
            message = (
                f"metric {metric} skipped: "
                f"{len(configs)} usable sample(s), "
                f"{distinct} distinct configuration(s)"
            )
            if explicit_metrics:
                raise ValueError(message)
            findings.append(message)
            continue

        trained[metric] = fit_metric_model(
            metric=metric,
            configs=configs,
            targets=targets,
            ridge_lambda=ridge_lambda,
        )

    if not trained:
        raise ValueError("no requested metric had enough data to train")

    observed_configs = tuple(
        sorted(
            {
                _row_config(row)
                for row in successful_rows
            },
            key=lambda config: config.label(),
        )
    )
    findings.insert(
        0,
        (
            "model is configuration-only: hardware/software provenance is "
            "guarded but not encoded as a predictive feature"
        ),
    )
    findings.append(
        "recommendation exploration uses distance from measured "
        "configurations as a heuristic, not calibrated uncertainty"
    )
    if len(hardware_signatures) > 1:
        findings.append(
            "mixed-hardware override enabled; predictions may conflate platform effects"
        )

    return PerformanceModelReport(
        campaign_ids=tuple(campaign.id for campaign in campaign_list),
        metrics=trained,
        observed_configs=observed_configs,
        hardware_signatures=hardware_signatures,
        ridge_lambda=ridge_lambda,
        findings=findings,
    )
