from adaptive_gpu_orchestrator.autopilot import (
    AutopilotPolicy,
    AutopilotStore,
    decide_autopilot,
)
from adaptive_gpu_orchestrator.controller import SchedulerBackend
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)
from adaptive_gpu_orchestrator.validation import ValidationPolicy


def make_optimization(identifier, tp, throughput):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=100.0,
            energy_joules=1000.0,
            memory_pressure_pct=50.0,
        ),
        score=1.0,
    )
    return OptimizationResult(
        id=identifier,
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
    )


def test_autopilot_store_round_trip(tmp_path):
    baseline = make_optimization("baseline", 1, 100.0)
    candidate = make_optimization("candidate", 2, 120.0)
    report = decide_autopilot(
        baseline=baseline,
        candidate=candidate,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=2,
        policy=AutopilotPolicy(
            validation=ValidationPolicy(min_throughput_gain_pct=5)
        ),
    )

    store = AutopilotStore(tmp_path)
    path = store.save(report)
    loaded = store.load(report.id)

    assert path.exists()
    assert loaded.id == report.id
    assert loaded.decision == report.decision
    assert loaded.policy == report.policy
    assert loaded.validation.decision == report.validation.decision
    assert loaded.plan_diff == report.plan_diff
    assert loaded.rollback.target_config_label == report.rollback.target_config_label
