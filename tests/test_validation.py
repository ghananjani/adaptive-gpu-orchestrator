from adaptive_gpu_orchestrator.optimizer import ParallelismConfig, TrialMetrics, TrialResult
from adaptive_gpu_orchestrator.validation import (
    ValidationDecision,
    ValidationPolicy,
    validate_trials,
)


def make_trial(
    *,
    tp=1,
    throughput=1000.0,
    latency=100.0,
    energy=1000.0,
    memory=50.0,
    exit_code=0,
):
    return TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            energy_joules=energy,
            memory_pressure_pct=memory,
            exit_code=exit_code,
        ),
    )


def test_validation_accepts_candidate_inside_regression_budget():
    baseline = make_trial()
    candidate = make_trial(
        tp=2,
        throughput=1100,
        latency=105,
        energy=1020,
        memory=51,
    )
    policy = ValidationPolicy(
        min_throughput_gain_pct=5,
        max_p99_latency_regression_pct=10,
        max_energy_regression_pct=5,
        max_memory_pressure_regression_pct=5,
    )

    report = validate_trials(baseline, candidate, policy)

    assert report.decision is ValidationDecision.ACCEPT
    assert report.accepted is True
    assert report.rollback_recommended is False
    assert report.deltas["throughput_per_s"].percent_change == 10


def test_validation_rejects_latency_regression_and_recommends_rollback():
    baseline = make_trial()
    candidate = make_trial(tp=2, throughput=1200, latency=125)
    policy = ValidationPolicy(
        min_throughput_gain_pct=5,
        max_p99_latency_regression_pct=5,
    )

    report = validate_trials(baseline, candidate, policy)

    assert report.decision is ValidationDecision.REJECT
    assert report.rollback_recommended is True
    assert any("p99 latency regression" in reason for reason in report.reasons)


def test_validation_rejects_missing_required_metric():
    baseline = make_trial()
    candidate = TrialResult(
        config=ParallelismConfig(tensor_parallel=2),
        metrics=TrialMetrics(throughput_per_s=1100),
    )

    report = validate_trials(
        baseline,
        candidate,
        ValidationPolicy(max_energy_regression_pct=5),
    )

    assert report.decision is ValidationDecision.REJECT
    assert any("energy metrics are required" in reason for reason in report.reasons)


def test_validation_rejects_failed_candidate():
    baseline = make_trial()
    candidate = make_trial(tp=2, exit_code=7)

    report = validate_trials(
        baseline,
        candidate,
        ValidationPolicy(min_throughput_gain_pct=0),
    )

    assert report.decision is ValidationDecision.REJECT
    assert any("candidate benchmark exited with code 7" in reason for reason in report.reasons)
