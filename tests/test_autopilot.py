import pytest

from adaptive_gpu_orchestrator.autopilot import (
    AutopilotDecision,
    AutopilotPolicy,
    decide_autopilot,
)
from adaptive_gpu_orchestrator.controller import DeploymentPolicy, SchedulerBackend
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)
from adaptive_gpu_orchestrator.studies import StudyGroupSummary, StudyReport
from adaptive_gpu_orchestrator.timeseries import TimeSeriesReport
from adaptive_gpu_orchestrator.validation import ValidationPolicy


def optimization(identifier, *, tp, throughput, latency, energy, memory):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            energy_joules=energy,
            memory_pressure_pct=memory,
            exit_code=0,
        ),
        score=1.0,
    )
    return OptimizationResult(
        id=identifier,
        objective=OptimizationObjective.BALANCED,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
        baseline=trial,
    )


def policy(**kwargs):
    return AutopilotPolicy(
        validation=ValidationPolicy(
            min_throughput_gain_pct=10,
            max_p99_latency_regression_pct=5,
            max_energy_regression_pct=5,
            max_memory_pressure_regression_pct=20,
        ),
        **kwargs,
    )


def baseline_and_candidate():
    baseline = optimization(
        "baseline",
        tp=1,
        throughput=100,
        latency=100,
        energy=1000,
        memory=50,
    )
    candidate = optimization(
        "candidate",
        tp=2,
        throughput=120,
        latency=95,
        energy=950,
        memory=55,
    )
    return baseline, candidate


def test_autopilot_promotes_candidate_when_all_configured_gates_pass():
    baseline, candidate = baseline_and_candidate()

    report = decide_autopilot(
        baseline=baseline,
        candidate=candidate,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=4,
        policy=policy(),
    )

    assert report.decision is AutopilotDecision.PROMOTE
    assert report.promotable is True
    assert report.validation.accepted is True
    assert report.plan_diff["changed"] is True
    assert report.rollback.target_config_label == baseline.best.config.label()
    assert report.rollback.baseline_plan["recommend_only"] is True


def test_autopilot_rejects_explicit_validation_failure():
    baseline, _ = baseline_and_candidate()
    candidate = optimization(
        "candidate",
        tp=2,
        throughput=105,
        latency=95,
        energy=950,
        memory=55,
    )

    report = decide_autopilot(
        baseline=baseline,
        candidate=candidate,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=4,
        policy=policy(),
    )

    assert report.decision is AutopilotDecision.REJECT
    assert report.promotable is False
    assert any("throughput gain" in reason for reason in report.reasons)
    assert report.rollback.trigger_reasons


def test_autopilot_holds_when_required_study_has_too_few_replicates():
    baseline, candidate = baseline_and_candidate()
    study = StudyReport(
        id="study1",
        groups=[
            StudyGroupSummary(
                name="baseline",
                optimization_ids=("baseline",),
                metrics={},
                config_counts={baseline.best.config.label(): 1},
                dominant_config=baseline.best.config.label(),
                config_stability_fraction=1.0,
            ),
            StudyGroupSummary(
                name="candidate",
                optimization_ids=("candidate", "candidate-rep2"),
                metrics={},
                config_counts={candidate.best.config.label(): 2},
                dominant_config=candidate.best.config.label(),
                config_stability_fraction=1.0,
            ),
        ],
    )

    report = decide_autopilot(
        baseline=baseline,
        candidate=candidate,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=4,
        policy=policy(require_study=True, min_study_replicates=3),
        study=study,
    )

    assert report.decision is AutopilotDecision.HOLD
    assert any("2 replicates" in reason for reason in report.reasons)
    assert report.study_id == "study1"


def test_autopilot_holds_on_required_timeseries_straggler_signal():
    baseline, candidate = baseline_and_candidate()
    timeseries = TimeSeriesReport(
        id="ts1",
        profile_id="profile1",
        straggler_gpu_indices=[1],
    )

    report = decide_autopilot(
        baseline=baseline,
        candidate=candidate,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=4,
        policy=policy(require_timeseries=True),
        timeseries=timeseries,
    )

    assert report.decision is AutopilotDecision.HOLD
    assert any("utilization stragglers" in reason for reason in report.reasons)
    assert report.timeseries_id == "ts1"


def test_autopilot_refuses_mutating_deployment_policy():
    baseline, candidate = baseline_and_candidate()

    with pytest.raises(ValueError, match="mutating deployment policy"):
        decide_autopilot(
            baseline=baseline,
            candidate=candidate,
            backend=SchedulerBackend.SLURM,
            command=["python", "train.py"],
            available_gpus=4,
            policy=policy(),
            deployment_policy=DeploymentPolicy(recommend_only=False),
        )
