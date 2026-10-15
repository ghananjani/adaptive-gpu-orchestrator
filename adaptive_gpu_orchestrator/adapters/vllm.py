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


class VLLMMetricsAdapter:
    """Normalize common vLLM serving benchmark metrics."""

    name = "vllm"

    _extra_keys = (
        "completed",
        "duration",
        "request_throughput",
        "request_goodput",
        "input_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p90_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p90_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p90_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "p99_e2el_ms",
        "total_input_tokens",
        "total_output_tokens",
    )

    def parse(self, payload: dict[str, Any]) -> TrialMetrics:
        throughput = _first_numeric(
            payload,
            "total_token_throughput",
            "output_throughput",
            "request_throughput",
            "throughput_per_s",
        )
        latency = _first_numeric(
            payload,
            "p99_e2el_ms",
            "p99_latency_ms",
            "p99_ttft_ms",
        )
        duration = _first_numeric(payload, "duration", "duration_s")

        return TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            energy_joules=_first_numeric(payload, "energy_joules"),
            gpu_util_pct=_first_numeric(payload, "gpu_util_pct"),
            memory_pressure_pct=_first_numeric(payload, "memory_pressure_pct"),
            duration_s=duration,
            exit_code=int(payload.get("exit_code", 0)),
            extras=numeric_extras(payload, keys=self._extra_keys),
        )
