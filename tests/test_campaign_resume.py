import pytest

from adaptive_gpu_orchestrator.campaigns import (
    CampaignProvenance,
    CampaignSpec,
    resume_campaign,
    run_campaign,
)
from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationStore,
    ParallelismConfig,
    TrialMetrics,
)
from adaptive_gpu_orchestrator.studies import StudyStore


class Runner:
    def __init__(self, *, fail=False, order=None):
        self.fail = fail
        self.order = order

    def run(self, config):
        if self.order is not None:
            self.order.append(config.label())
        if self.fail:
            raise OSError("synthetic benchmark failure")
        return TrialMetrics(
            throughput_per_s=100.0 + config.tensor_parallel,
            p99_latency_ms=100.0,
            energy_joules=1000.0,
        )


def provenance(model="H100", package_version="2.9.0"):
    return CampaignProvenance(
        runtime=RuntimeProvenance(
            python_version="3.12.0",
            implementation="cpython",
            platform="Linux-test",
            machine="x86_64",
        ),
        cluster_snapshot={
            "hostname": "gpu-node",
            "driver_version": "580.12",
            "cuda_version": "13.0",
            "gpus": [
                {
                    "name": model,
                    "memory_total_mib": 81559,
                    "compute_capability": "9.0",
                    "utilization_gpu_pct": 55.0,
                }
            ],
        },
        package_versions={"torch": package_version},
    )


def spec(*, randomized=False, seed=None):
    return CampaignSpec(
        name="resume-test",
        group="h100",
        repeats=3,
        available_gpus=3,
        candidates=(
            ParallelismConfig(tensor_parallel=1),
            ParallelismConfig(tensor_parallel=2),
            ParallelismConfig(tensor_parallel=3),
        ),
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        adapter="generic",
        benchmark_command=("python", "bench.py"),
        randomize_candidates=randomized,
        random_seed=seed,
    )


def test_resume_only_reruns_incomplete_replicates(tmp_path):
    optimizations = OptimizationStore(tmp_path / "optimizations")
    studies = StudyStore(tmp_path / "studies")
    first_calls = []

    def first_factory(replicate, env):
        first_calls.append(replicate)
        return Runner(fail=replicate == 2)

    report = run_campaign(
        spec=spec(),
        provenance=provenance(),
        runner_factory=first_factory,
        optimization_store=optimizations,
        study_store=studies,
        campaign_id="campaign1",
    )
    completed_before = {
        run.replicate: run.optimization_id
        for run in report.runs
        if run.status == "completed"
    }
    assert report.pending_replicates == (2,)

    resume_calls = []

    def resume_factory(replicate, env):
        resume_calls.append(replicate)
        return Runner()

    resumed = resume_campaign(
        report,
        runner_factory=resume_factory,
        optimization_store=optimizations,
        study_store=studies,
        current_provenance=provenance(),
    )

    assert first_calls == [1, 2, 3]
    assert resume_calls == [2]
    assert resumed.successful_replicates == 3
    assert resumed.pending_replicates == ()
    assert resumed.resume_count == 1
    assert len(resumed.resume_provenance) == 1
    assert resumed.manifest_sha256 is not None
    assert resumed.study_id is not None

    for replicate, identifier in completed_before.items():
        run = next(item for item in resumed.runs if item.replicate == replicate)
        assert run.optimization_id == identifier


def test_resume_rejects_changed_static_provenance(tmp_path):
    optimizations = OptimizationStore(tmp_path / "optimizations")
    studies = StudyStore(tmp_path / "studies")

    report = run_campaign(
        spec=spec(),
        provenance=provenance("H100"),
        runner_factory=lambda replicate, env: Runner(fail=replicate == 2),
        optimization_store=optimizations,
        study_store=studies,
    )

    with pytest.raises(ValueError, match="provenance changed"):
        resume_campaign(
            report,
            runner_factory=lambda replicate, env: Runner(),
            optimization_store=optimizations,
            study_store=studies,
            current_provenance=provenance("H200"),
        )


def test_resume_can_explicitly_record_provenance_change(tmp_path):
    optimizations = OptimizationStore(tmp_path / "optimizations")
    studies = StudyStore(tmp_path / "studies")

    report = run_campaign(
        spec=spec(),
        provenance=provenance("H100"),
        runner_factory=lambda replicate, env: Runner(fail=replicate == 2),
        optimization_store=optimizations,
        study_store=studies,
    )
    changed = provenance("H200")

    resumed = resume_campaign(
        report,
        runner_factory=lambda replicate, env: Runner(),
        optimization_store=optimizations,
        study_store=studies,
        current_provenance=changed,
        allow_provenance_change=True,
    )

    assert resumed.resume_provenance[-1] == changed
    assert resumed.successful_replicates == 3


def test_candidate_randomization_is_deterministic_and_audited(tmp_path):
    def run_once(root):
        optimizations = OptimizationStore(root / "optimizations")
        studies = StudyStore(root / "studies")
        observed = []

        report = run_campaign(
            spec=spec(randomized=True, seed=1234),
            provenance=provenance(),
            runner_factory=lambda replicate, env: Runner(order=observed),
            optimization_store=optimizations,
            study_store=studies,
        )
        return report, observed

    left, left_observed = run_once(tmp_path / "left")
    right, right_observed = run_once(tmp_path / "right")

    assert left_observed == right_observed
    assert [run.candidate_order for run in left.runs] == [
        run.candidate_order
        for run in right.runs
    ]
    assert all(len(run.candidate_order) == 3 for run in left.runs)
