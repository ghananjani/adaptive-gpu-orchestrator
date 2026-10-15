from __future__ import annotations

import math
from collections import Counter
from statistics import mean, stdev

from adaptive_gpu_orchestrator.optimizer import OptimizationResult, TrialMetrics

from .models import (
    MetricComparison,
    MetricSummary,
    ParetoPoint,
    StudyGroupSummary,
    StudyReport,
)

_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

_LOWER_IS_BETTER = {
    "p99_latency_ms",
    "energy_joules",
    "memory_pressure_pct",
    "duration_s",
}


def _metrics(metrics: TrialMetrics) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in (
        "throughput_per_s",
        "p99_latency_ms",
        "energy_joules",
        "gpu_util_pct",
        "memory_pressure_pct",
        "duration_s",
    ):
        value = getattr(metrics, name)
        if value is not None:
            values[name] = float(value)

    values.update(
        {
            f"extras.{name}": float(value)
            for name, value in metrics.extras.items()
        }
    )
    return values


def _summary(metric: str, values: list[float]) -> MetricSummary:
    if not values:
        raise ValueError("metric summary requires at least one value")

    average = mean(values)
    if len(values) == 1:
        return MetricSummary(
            metric=metric,
            n=1,
            mean=average,
            minimum=values[0],
            maximum=values[0],
        )

    sample_stdev = stdev(values)
    df = len(values) - 1
    critical = _T_CRITICAL_95.get(df, 1.96)
    margin = critical * sample_stdev / math.sqrt(len(values))
    return MetricSummary(
        metric=metric,
        n=len(values),
        mean=average,
        minimum=min(values),
        maximum=max(values),
        stdev=sample_stdev,
        ci95_low=average - margin,
        ci95_high=average + margin,
    )


def _cohens_d(
    baseline: MetricSummary,
    candidate: MetricSummary,
) -> float | None:
    if (
        baseline.n < 2
        or candidate.n < 2
        or baseline.stdev is None
        or candidate.stdev is None
    ):
        return None

    pooled_denominator = baseline.n + candidate.n - 2
    if pooled_denominator <= 0:
        return None

    pooled_variance = (
        (baseline.n - 1) * baseline.stdev**2
        + (candidate.n - 1) * candidate.stdev**2
    ) / pooled_denominator
    if pooled_variance <= 0:
        return None

    return (
        candidate.mean - baseline.mean
    ) / math.sqrt(pooled_variance)


def _group_summary(
    name: str,
    results: list[OptimizationResult],
) -> StudyGroupSummary:
    if not results:
        raise ValueError(f"study group '{name}' has no runs")

    by_metric: dict[str, list[float]] = {}
    config_counts: Counter[str] = Counter()
    optimization_ids: list[str] = []

    for result in results:
        if result.best is None:
            raise ValueError(
                f"optimization '{result.id}' has no feasible best trial"
            )
        optimization_ids.append(result.id)
        config_counts[result.best.config.label()] += 1
        for metric, value in _metrics(result.best.metrics).items():
            by_metric.setdefault(metric, []).append(value)

    metrics = {
        metric: _summary(metric, values)
        for metric, values in sorted(by_metric.items())
    }

    dominant_config = None
    stability = None
    if config_counts:
        dominant_config, count = config_counts.most_common(1)[0]
        stability = count / len(results)

    return StudyGroupSummary(
        name=name,
        optimization_ids=tuple(optimization_ids),
        metrics=metrics,
        config_counts=dict(config_counts),
        dominant_config=dominant_config,
        config_stability_fraction=stability,
    )


def _comparison(
    baseline: StudyGroupSummary,
    candidate: StudyGroupSummary,
) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    common = sorted(set(baseline.metrics) & set(candidate.metrics))

    for metric in common:
        base = baseline.metrics[metric]
        cand = candidate.metrics[metric]
        absolute = cand.mean - base.mean
        percent = None
        if base.mean != 0:
            percent = 100.0 * absolute / base.mean

        improvement = None
        if percent is not None:
            improvement = -percent if metric in _LOWER_IS_BETTER else percent

        comparisons.append(
            MetricComparison(
                metric=metric,
                baseline_group=baseline.name,
                candidate_group=candidate.name,
                baseline_mean=base.mean,
                candidate_mean=cand.mean,
                absolute_change=absolute,
                percent_change=percent,
                relative_improvement_pct=improvement,
                cohens_d=_cohens_d(base, cand),
            )
        )
    return comparisons


def _pareto_frontier(
    groups: dict[str, list[OptimizationResult]],
) -> list[ParetoPoint]:
    points: list[ParetoPoint] = []
    raw: list[tuple[str, str, str, dict[str, float]]] = []

    for group, results in groups.items():
        for result in results:
            if result.best is None:
                continue
            metrics = _metrics(result.best.metrics)
            if (
                "throughput_per_s" in metrics
                and "p99_latency_ms" in metrics
                and "energy_joules" in metrics
            ):
                raw.append(
                    (
                        result.id,
                        group,
                        result.best.config.label(),
                        {
                            "throughput_per_s": metrics["throughput_per_s"],
                            "p99_latency_ms": metrics["p99_latency_ms"],
                            "energy_joules": metrics["energy_joules"],
                        },
                    )
                )

    for index, candidate in enumerate(raw):
        candidate_metrics = candidate[3]
        dominated = False
        for other_index, other in enumerate(raw):
            if index == other_index:
                continue
            other_metrics = other[3]

            no_worse = (
                other_metrics["throughput_per_s"]
                >= candidate_metrics["throughput_per_s"]
                and other_metrics["p99_latency_ms"]
                <= candidate_metrics["p99_latency_ms"]
                and other_metrics["energy_joules"]
                <= candidate_metrics["energy_joules"]
            )
            strictly_better = (
                other_metrics["throughput_per_s"]
                > candidate_metrics["throughput_per_s"]
                or other_metrics["p99_latency_ms"]
                < candidate_metrics["p99_latency_ms"]
                or other_metrics["energy_joules"]
                < candidate_metrics["energy_joules"]
            )
            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            points.append(
                ParetoPoint(
                    optimization_id=candidate[0],
                    group=candidate[1],
                    config_label=candidate[2],
                    metrics=candidate_metrics,
                )
            )

    return points


def analyze_study(
    groups: dict[str, list[OptimizationResult]],
) -> StudyReport:
    if not groups:
        raise ValueError("study requires at least one group")

    summaries = [
        _group_summary(name, results)
        for name, results in groups.items()
    ]

    comparisons: list[MetricComparison] = []
    baseline = summaries[0]
    for candidate in summaries[1:]:
        comparisons.extend(_comparison(baseline, candidate))

    findings: list[str] = []
    for group in summaries:
        if len(group.optimization_ids) < 3:
            findings.append(
                f"Group '{group.name}' has fewer than three replicates; "
                "confidence intervals and effect sizes are exploratory."
            )
        if (
            group.config_stability_fraction is not None
            and group.config_stability_fraction < 0.75
        ):
            findings.append(
                f"Group '{group.name}' selected multiple best configurations; "
                f"dominant-config stability is "
                f"{100.0 * group.config_stability_fraction:.1f}%."
            )

    pareto = _pareto_frontier(groups)
    if not pareto:
        findings.append(
            "Pareto frontier unavailable because runs do not all expose "
            "throughput, p99 latency, and energy."
        )

    return StudyReport(
        groups=summaries,
        comparisons=comparisons,
        pareto_frontier=pareto,
        findings=findings,
    )
