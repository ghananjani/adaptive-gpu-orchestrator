from __future__ import annotations

from adaptive_gpu_orchestrator.optimizer.trials import (
    OptimizationConstraints,
    OptimizationObjective,
    TrialResult,
)


def apply_constraints(
    trial: TrialResult,
    constraints: OptimizationConstraints,
) -> TrialResult:
    reasons: list[str] = []
    metrics = trial.metrics

    if metrics.exit_code != 0:
        reasons.append(f"benchmark exited with code {metrics.exit_code}")

    if constraints.max_p99_latency_ms is not None:
        if metrics.p99_latency_ms is None:
            reasons.append("p99 latency metric is required by the configured constraint")
        elif metrics.p99_latency_ms > constraints.max_p99_latency_ms:
            reasons.append(
                f"p99 latency {metrics.p99_latency_ms:.3f} ms exceeds "
                f"{constraints.max_p99_latency_ms:.3f} ms"
            )

    if constraints.max_energy_joules is not None:
        if metrics.energy_joules is None:
            reasons.append("energy metric is required by the configured constraint")
        elif metrics.energy_joules > constraints.max_energy_joules:
            reasons.append(
                f"energy {metrics.energy_joules:.3f} J exceeds "
                f"{constraints.max_energy_joules:.3f} J"
            )

    if constraints.min_throughput_per_s is not None:
        if metrics.throughput_per_s is None:
            reasons.append("throughput metric is required by the configured constraint")
        elif metrics.throughput_per_s < constraints.min_throughput_per_s:
            reasons.append(
                f"throughput {metrics.throughput_per_s:.3f}/s is below "
                f"{constraints.min_throughput_per_s:.3f}/s"
            )

    if constraints.max_memory_pressure_pct is not None:
        if metrics.memory_pressure_pct is None:
            reasons.append("memory-pressure metric is required by the configured constraint")
        elif metrics.memory_pressure_pct > constraints.max_memory_pressure_pct:
            reasons.append(
                f"memory pressure {metrics.memory_pressure_pct:.3f}% exceeds "
                f"{constraints.max_memory_pressure_pct:.3f}%"
            )

    trial.feasible = not reasons
    trial.rejection_reasons = reasons
    return trial


def _ratio(value: float | None, reference: float | None, maximize: bool) -> float | None:
    if value is None or reference is None or value <= 0 or reference <= 0:
        return None
    if maximize:
        return value / reference
    return reference / value


def score_trials(
    trials: list[TrialResult],
    objective: OptimizationObjective,
) -> list[TrialResult]:
    feasible = [trial for trial in trials if trial.feasible]
    if not feasible:
        return trials

    throughputs = [
        trial.metrics.throughput_per_s
        for trial in feasible
        if trial.metrics.throughput_per_s is not None and trial.metrics.throughput_per_s > 0
    ]
    latencies = [
        trial.metrics.p99_latency_ms
        for trial in feasible
        if trial.metrics.p99_latency_ms is not None and trial.metrics.p99_latency_ms > 0
    ]
    energies = [
        trial.metrics.energy_joules
        for trial in feasible
        if trial.metrics.energy_joules is not None and trial.metrics.energy_joules > 0
    ]

    max_throughput = max(throughputs, default=None)
    min_latency = min(latencies, default=None)
    min_energy = min(energies, default=None)

    for trial in feasible:
        throughput_score = _ratio(
            trial.metrics.throughput_per_s,
            max_throughput,
            maximize=True,
        )
        latency_score = _ratio(
            trial.metrics.p99_latency_ms,
            min_latency,
            maximize=False,
        )
        energy_score = _ratio(
            trial.metrics.energy_joules,
            min_energy,
            maximize=False,
        )

        if objective is OptimizationObjective.THROUGHPUT:
            trial.score = throughput_score
        elif objective is OptimizationObjective.LATENCY:
            trial.score = latency_score
        elif objective is OptimizationObjective.ENERGY:
            trial.score = energy_score
        else:
            weighted = [
                (throughput_score, 0.5),
                (latency_score, 0.3),
                (energy_score, 0.2),
            ]
            present = [(value, weight) for value, weight in weighted if value is not None]
            total_weight = sum(weight for _, weight in present)
            trial.score = (
                sum(value * weight for value, weight in present) / total_weight
                if total_weight
                else None
            )

    return trials
