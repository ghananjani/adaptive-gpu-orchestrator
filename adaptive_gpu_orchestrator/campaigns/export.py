from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from adaptive_gpu_orchestrator.optimizer import OptimizationStore
from .models import CampaignReport


_BASE_COLUMNS = (
    "campaign_id",
    "campaign_name",
    "campaign_group",
    "replicate",
    "optimization_id",
    "trial_index",
    "selected_best",
    "feasible",
    "config_label",
    "tensor_parallel",
    "pipeline_parallel",
    "data_parallel",
    "batch_size",
    "concurrency",
    "throughput_per_s",
    "p99_latency_ms",
    "energy_joules",
    "gpu_util_pct",
    "memory_pressure_pct",
    "duration_s",
    "exit_code",
    "driver_version",
    "cuda_runtime_version",
    "cuda_toolkit_version",
    "nccl_version",
    "container_image",
    "container_image_digest",
    "gpu_count",
    "gpu_models",
    "manifest_sha256",
    "nccl_mean_algbw_gbps",
    "nccl_peak_algbw_gbps",
    "nccl_mean_busbw_gbps",
    "nccl_peak_busbw_gbps",
)


def campaign_trial_rows(
    report: CampaignReport,
    *,
    optimization_store: OptimizationStore,
) -> list[dict[str, Any]]:
    snapshot = report.provenance.cluster_snapshot
    gpu_rows = snapshot.get("gpus", [])
    gpu_models = sorted(
        {
            str(item.get("name"))
            for item in gpu_rows
            if item.get("name")
        }
    )

    nccl_summary = (
        report.nccl_baseline.summary
        if report.nccl_baseline is not None
        else {}
    )

    common = {
        "campaign_id": report.id,
        "campaign_name": report.spec.name,
        "campaign_group": report.spec.group,
        "driver_version": snapshot.get("driver_version"),
        "cuda_runtime_version": snapshot.get("cuda_version"),
        "cuda_toolkit_version": report.provenance.cuda_toolkit_version,
        "nccl_version": report.provenance.nccl_version,
        "container_image": report.provenance.container_image,
        "container_image_digest": report.provenance.container_image_digest,
        "gpu_count": len(gpu_rows),
        "gpu_models": "|".join(gpu_models),
        "manifest_sha256": report.manifest_sha256,
        "nccl_mean_algbw_gbps": nccl_summary.get("mean_algbw_gbps"),
        "nccl_peak_algbw_gbps": nccl_summary.get("peak_algbw_gbps"),
        "nccl_mean_busbw_gbps": nccl_summary.get("mean_busbw_gbps"),
        "nccl_peak_busbw_gbps": nccl_summary.get("peak_busbw_gbps"),
    }

    rows: list[dict[str, Any]] = []
    for run in sorted(report.runs, key=lambda item: item.replicate):
        if run.optimization_id is None:
            continue
        optimization = optimization_store.load(run.optimization_id)
        best = optimization.best

        for index, trial in enumerate(optimization.trials, start=1):
            config = trial.config
            metrics = trial.metrics
            row = {
                **common,
                "replicate": run.replicate,
                "optimization_id": optimization.id,
                "trial_index": index,
                "selected_best": (
                    best is not None
                    and best.config == trial.config
                    and best.metrics == trial.metrics
                ),
                "feasible": trial.feasible,
                "config_label": config.label(),
                "tensor_parallel": config.tensor_parallel,
                "pipeline_parallel": config.pipeline_parallel,
                "data_parallel": config.data_parallel,
                "batch_size": config.batch_size,
                "concurrency": config.concurrency,
                "throughput_per_s": metrics.throughput_per_s,
                "p99_latency_ms": metrics.p99_latency_ms,
                "energy_joules": metrics.energy_joules,
                "gpu_util_pct": metrics.gpu_util_pct,
                "memory_pressure_pct": metrics.memory_pressure_pct,
                "duration_s": metrics.duration_s,
                "exit_code": metrics.exit_code,
            }
            for name, value in sorted(metrics.extras.items()):
                row[f"extra_{name}"] = value
            rows.append(row)
    return rows


def write_campaign_export(
    report: CampaignReport,
    *,
    optimization_store: OptimizationStore,
    output: str | Path,
    format: str = "csv",
) -> Path:
    path = Path(output)
    rows = campaign_trial_rows(
        report,
        optimization_store=optimization_store,
    )
    if not rows:
        raise ValueError("campaign has no optimization trials to export")

    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
        return path

    if format != "csv":
        raise ValueError("campaign export format must be csv or jsonl")

    extra_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in _BASE_COLUMNS
        }
    )
    columns = [*_BASE_COLUMNS, *extra_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path
