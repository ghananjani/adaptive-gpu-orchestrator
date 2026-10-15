from adaptive_gpu_orchestrator.optimizer import ParallelismConfig, TrialMetrics, TrialResult
from adaptive_gpu_orchestrator.validation import (
    ValidationPolicy,
    ValidationStore,
    validate_trials,
)


def test_validation_store_round_trip(tmp_path):
    baseline = TrialResult(
        config=ParallelismConfig(tensor_parallel=1),
        metrics=TrialMetrics(throughput_per_s=1000, p99_latency_ms=100),
    )
    candidate = TrialResult(
        config=ParallelismConfig(tensor_parallel=2),
        metrics=TrialMetrics(throughput_per_s=1200, p99_latency_ms=102),
    )
    report = validate_trials(
        baseline,
        candidate,
        ValidationPolicy(
            min_throughput_gain_pct=5,
            max_p99_latency_regression_pct=5,
        ),
    )

    store = ValidationStore(tmp_path)
    path = store.save(report)
    loaded = store.load(report.id)

    assert path.exists()
    assert loaded.id == report.id
    assert loaded.decision == report.decision
    assert loaded.candidate_config == candidate.config
    assert loaded.deltas["throughput_per_s"].percent_change == 20


def test_validation_store_lists_reports(tmp_path):
    store = ValidationStore(tmp_path)
    for tp in (2, 4):
        baseline = TrialResult(
            config=ParallelismConfig(tensor_parallel=1),
            metrics=TrialMetrics(throughput_per_s=1000),
        )
        candidate = TrialResult(
            config=ParallelismConfig(tensor_parallel=tp),
            metrics=TrialMetrics(throughput_per_s=1100),
        )
        store.save(
            validate_trials(
                baseline,
                candidate,
                ValidationPolicy(min_throughput_gain_pct=0),
            )
        )

    assert len(store.list()) == 2
