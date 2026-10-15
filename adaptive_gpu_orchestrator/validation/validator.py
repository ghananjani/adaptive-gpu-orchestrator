from __future__ import annotations

from uuid import uuid4

from adaptive_gpu_orchestrator.optimizer import TrialResult

from .models import (
    MetricDelta,
    ValidationDecision,
    ValidationPolicy,
    ValidationReport,
)


def _delta(baseline: float | None, candidate: float | None) -> MetricDelta | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return MetricDelta(
        baseline=float(baseline),
        candidate=float(candidate),
        percent_change=100.0 * (candidate - baseline) / baseline,
    )


def validate_trials(
    baseline: TrialResult,
    candidate: TrialResult,
    policy: ValidationPolicy,
) -> ValidationReport:
    reasons: list[str] = []
    deltas: dict[str, MetricDelta] = {}

    if baseline.metrics.exit_code != 0:
        reasons.append(
            f"baseline benchmark exited with code {baseline.metrics.exit_code}"
        )
    if candidate.metrics.exit_code != 0:
        reasons.append(
            f"candidate benchmark exited with code {candidate.metrics.exit_code}"
        )

    metric_pairs = {
        "throughput_per_s": (
            baseline.metrics.throughput_per_s,
            candidate.metrics.throughput_per_s,
        ),
        "p99_latency_ms": (
            baseline.metrics.p99_latency_ms,
            candidate.metrics.p99_latency_ms,
        ),
        "energy_joules": (
            baseline.metrics.energy_joules,
            candidate.metrics.energy_joules,
        ),
        "memory_pressure_pct": (
            baseline.metrics.memory_pressure_pct,
            candidate.metrics.memory_pressure_pct,
        ),
        "gpu_util_pct": (
            baseline.metrics.gpu_util_pct,
            candidate.metrics.gpu_util_pct,
        ),
    }
    for name, (baseline_value, candidate_value) in metric_pairs.items():
        delta = _delta(baseline_value, candidate_value)
        if delta is not None:
            deltas[name] = delta

    def required_delta(name: str, message: str) -> MetricDelta | None:
        delta = deltas.get(name)
        if delta is None:
            reasons.append(message)
        return delta

    if policy.min_throughput_gain_pct is not None:
        delta = required_delta(
            "throughput_per_s",
            "throughput metrics are required by the validation policy",
        )
        if delta is not None and delta.percent_change < policy.min_throughput_gain_pct:
            reasons.append(
                f"throughput gain {delta.percent_change:.3f}% is below required "
                f"{policy.min_throughput_gain_pct:.3f}%"
            )

    if policy.max_p99_latency_regression_pct is not None:
        delta = required_delta(
            "p99_latency_ms",
            "p99 latency metrics are required by the validation policy",
        )
        if (
            delta is not None
            and delta.percent_change > policy.max_p99_latency_regression_pct
        ):
            reasons.append(
                f"p99 latency regression {delta.percent_change:.3f}% exceeds allowed "
                f"{policy.max_p99_latency_regression_pct:.3f}%"
            )

    if policy.max_energy_regression_pct is not None:
        delta = required_delta(
            "energy_joules",
            "energy metrics are required by the validation policy",
        )
        if delta is not None and delta.percent_change > policy.max_energy_regression_pct:
            reasons.append(
                f"energy regression {delta.percent_change:.3f}% exceeds allowed "
                f"{policy.max_energy_regression_pct:.3f}%"
            )

    if policy.max_memory_pressure_regression_pct is not None:
        delta = required_delta(
            "memory_pressure_pct",
            "memory-pressure metrics are required by the validation policy",
        )
        if (
            delta is not None
            and delta.percent_change > policy.max_memory_pressure_regression_pct
        ):
            reasons.append(
                f"memory-pressure regression {delta.percent_change:.3f}% exceeds allowed "
                f"{policy.max_memory_pressure_regression_pct:.3f}%"
            )

    decision = ValidationDecision.REJECT if reasons else ValidationDecision.ACCEPT
    return ValidationReport(
        id=uuid4().hex[:12],
        decision=decision,
        policy=policy,
        baseline_config=baseline.config,
        candidate_config=candidate.config,
        deltas=deltas,
        reasons=reasons,
        rollback_recommended=(
            decision is ValidationDecision.REJECT and policy.rollback_on_reject
        ),
    )
