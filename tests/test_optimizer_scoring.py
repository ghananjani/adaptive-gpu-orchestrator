from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig
from adaptive_gpu_orchestrator.optimizer.scoring import apply_constraints, score_trials
from adaptive_gpu_orchestrator.optimizer.trials import (
    OptimizationConstraints,
    OptimizationObjective,
    TrialMetrics,
    TrialResult,
)


def trial(throughput, latency, energy, memory=60):
    return TrialResult(
        config=ParallelismConfig(),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            energy_joules=energy,
            memory_pressure_pct=memory,
        ),
    )


def test_constraints_reject_slo_violation():
    candidate = trial(1000, 120, 3000)
    apply_constraints(
        candidate,
        OptimizationConstraints(max_p99_latency_ms=100),
    )
    assert candidate.feasible is False
    assert candidate.rejection_reasons


def test_throughput_scoring_prefers_highest_throughput():
    trials = [
        trial(1000, 100, 3000),
        trial(1400, 130, 3500),
    ]
    score_trials(trials, OptimizationObjective.THROUGHPUT)
    assert trials[1].score > trials[0].score


def test_balanced_scoring_uses_multiple_metrics():
    trials = [
        trial(1000, 80, 2500),
        trial(1300, 160, 5000),
    ]
    score_trials(trials, OptimizationObjective.BALANCED)
    assert all(item.score is not None for item in trials)
    assert trials[0].score != trials[1].score
