from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)
from adaptive_gpu_orchestrator.studies import analyze_study


def optimization(
    identifier,
    *,
    throughput,
    latency,
    energy,
    tp=2,
):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            energy_joules=energy,
        ),
        score=1.0,
    )
    return OptimizationResult(
        id=identifier,
        objective=OptimizationObjective.BALANCED,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
    )


def test_study_summarizes_replicates_and_effect_sizes():
    baseline = [
        optimization("a1", throughput=90, latency=105, energy=1100, tp=1),
        optimization("a2", throughput=100, latency=100, energy=1000, tp=1),
        optimization("a3", throughput=110, latency=95, energy=900, tp=2),
    ]
    candidate = [
        optimization("b1", throughput=110, latency=92, energy=960, tp=2),
        optimization("b2", throughput=120, latency=90, energy=950, tp=2),
        optimization("b3", throughput=130, latency=88, energy=940, tp=2),
    ]

    report = analyze_study({"h100": baseline, "h200": candidate})

    h100 = report.groups[0]
    h200 = report.groups[1]

    assert h100.metrics["throughput_per_s"].mean == 100
    assert h100.metrics["throughput_per_s"].ci95_low is not None
    assert h100.config_stability_fraction == 2 / 3
    assert h200.config_stability_fraction == 1.0

    throughput = next(
        item
        for item in report.comparisons
        if item.metric == "throughput_per_s"
    )
    latency = next(
        item
        for item in report.comparisons
        if item.metric == "p99_latency_ms"
    )

    assert throughput.percent_change == 20.0
    assert throughput.relative_improvement_pct == 20.0
    assert throughput.cohens_d is not None
    assert latency.percent_change == -10.0
    assert latency.relative_improvement_pct == 10.0
    assert any("dominant-config stability" in item for item in report.findings)


def test_pareto_frontier_removes_dominated_run():
    slow = optimization(
        "slow",
        throughput=100,
        latency=100,
        energy=1000,
    )
    fast = optimization(
        "fast",
        throughput=120,
        latency=90,
        energy=900,
    )

    report = analyze_study({"baseline": [slow], "candidate": [fast]})

    assert [point.optimization_id for point in report.pareto_frontier] == [
        "fast"
    ]


def test_study_preserves_adapter_specific_metrics():
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=2),
        metrics=TrialMetrics(
            throughput_per_s=1000,
            extras={"p99_ttft_ms": 75.0},
        ),
        score=1.0,
    )
    result = OptimizationResult(
        id="vllm",
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
    )

    report = analyze_study({"serving": [result]})

    assert report.groups[0].metrics["extras.p99_ttft_ms"].mean == 75.0
