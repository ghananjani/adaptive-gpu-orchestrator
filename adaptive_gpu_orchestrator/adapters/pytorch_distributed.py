from __future__ import annotations

from typing import Any

from adaptive_gpu_orchestrator.optimizer.trials import TrialMetrics

from .base import numeric_extras


def _first_numeric(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class PyTorchDistributedMetricsAdapter:
    """Normalize distributed-training benchmark summaries."""

    name = "pytorch-distributed"

    _extra_keys = (
        "samples_per_s",
        "tokens_per_s",
        "sequences_per_s",
        "step_time_ms",
        "mean_step_time_ms",
        "p95_step_time_ms",
        "p99_step_time_ms",
        "compute_time_ms",
        "communication_time_ms",
        "all_reduce_time_ms",
        "forward_time_ms",
        "backward_time_ms",
        "optimizer_time_ms",
        "data_time_ms",
        "dataloader_time_ms",
        "tokens_per_gpu_per_s",
        "samples_per_gpu_per_s",
        "gradient_sync_time_ms",
    )

    def parse(self, payload: dict[str, Any]) -> TrialMetrics:
        throughput = _first_numeric(
            payload,
            "throughput_per_s",
            "tokens_per_s",
            "samples_per_s",
            "sequences_per_s",
        )
        latency = _first_numeric(
            payload,
            "p99_latency_ms",
            "p99_step_time_ms",
            "step_time_p99_ms",
        )

        extras = numeric_extras(
            payload,
            keys=self._extra_keys,
            prefixes=("phases",),
        )

        step_time = _first_numeric(
            payload,
            "mean_step_time_ms",
            "step_time_ms",
        )
        communication = _first_numeric(
            payload,
            "communication_time_ms",
            "all_reduce_time_ms",
            "gradient_sync_time_ms",
        )
        compute = _first_numeric(payload, "compute_time_ms")

        if step_time is not None and step_time > 0:
            if communication is not None:
                extras["communication_fraction_pct"] = 100.0 * communication / step_time
            if compute is not None:
                extras["compute_fraction_pct"] = 100.0 * compute / step_time

        return TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            energy_joules=_first_numeric(payload, "energy_joules"),
            gpu_util_pct=_first_numeric(payload, "gpu_util_pct"),
            memory_pressure_pct=_first_numeric(payload, "memory_pressure_pct"),
            duration_s=_first_numeric(payload, "duration_s", "duration"),
            exit_code=int(payload.get("exit_code", 0)),
            extras=extras,
        )
