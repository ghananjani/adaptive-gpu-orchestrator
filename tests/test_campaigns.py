from adaptive_gpu_orchestrator.campaigns import (
    CampaignProvenance,
    CampaignSpec,
    compare_campaigns,
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


class FakeRunner:
    def __init__(self, replicate, env, offset):
        self.replicate = replicate
        self.env = env
        self.offset = offset

    def run(self, config):
        return TrialMetrics(
            throughput_per_s=(
                100.0
                + self.offset
                + 10.0 * config.tensor_parallel
                + self.replicate
            ),
            p99_latency_ms=100.0 - config.tensor_parallel,
            energy_joules=1000.0 + config.tensor_parallel,
        )


def provenance(label):
    return CampaignProvenance(
        runtime=RuntimeProvenance(
            python_version="3.12.0",
            implementation="cpython",
            platform="Linux-test",
            machine="x86_64",
            tags={"hardware": label},
        ),
        cluster_snapshot={
            "hostname": "node-a",
            "driver_version": "580.0",
            "cuda_version": "13.0",
            "gpus": [{"name": label}],
            "topology": [],
            "warnings": [],
        },
        package_versions={"torch": "2.9.0"},
    )


def spec(group, repeats=3):
    return CampaignSpec(
        name=f"{group}-benchmark",
        group=group,
        repeats=repeats,
        available_gpus=2,
        candidates=(
            ParallelismConfig(tensor_parallel=1),
            ParallelismConfig(tensor_parallel=2),
        ),
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        adapter="generic",
        benchmark_command=("python", "benchmark.py"),
        tags={"hardware": group},
    )


def test_campaign_runs_replicates_persists_optimizations_and_builds_study(tmp_path):
    optimizations = OptimizationStore(tmp_path / "optimizations")
    studies = StudyStore(tmp_path / "studies")
    seen_env = []

    def factory(replicate, env):
        seen_env.append(dict(env))
        return FakeRunner(replicate, env, offset=0)

    report = run_campaign(
        spec=spec("h100"),
        provenance=provenance("H100"),
        runner_factory=factory,
        optimization_store=optimizations,
        study_store=studies,
        campaign_id="campaign123",
    )

    assert report.id == "campaign123"
    assert report.successful_replicates == 3
    assert len(report.completed_optimization_ids) == 3
    assert all(run.best_config == "tp2-pp1-dp1-b1-c1" for run in report.runs)
    assert report.study_id is not None
    assert studies.load(report.study_id).groups[0].name == "h100"
    assert len(optimizations.list()) == 3
    assert seen_env[0]["AGO_CAMPAIGN_ID"] == "campaign123"
    assert seen_env[0]["AGO_CAMPAIGN_GROUP"] == "h100"
    assert seen_env[2]["AGO_REPLICATE_INDEX"] == "3"
    assert seen_env[0]["AGO_CAMPAIGN_TAG_HARDWARE"] == "h100"


def test_compare_campaigns_reuses_completed_optimizer_runs(tmp_path):
    optimizations = OptimizationStore(tmp_path / "optimizations")
    studies = StudyStore(tmp_path / "studies")

    h100 = run_campaign(
        spec=spec("h100"),
        provenance=provenance("H100"),
        runner_factory=lambda replicate, env: FakeRunner(replicate, env, 0),
        optimization_store=optimizations,
        study_store=studies,
        campaign_id="h100-campaign",
    )
    h200 = run_campaign(
        spec=spec("h200"),
        provenance=provenance("H200"),
        runner_factory=lambda replicate, env: FakeRunner(replicate, env, 50),
        optimization_store=optimizations,
        study_store=studies,
        campaign_id="h200-campaign",
    )

    comparison = compare_campaigns(
        [h100, h200],
        optimization_store=optimizations,
        study_store=studies,
    )

    assert [group.name for group in comparison.groups] == ["h100", "h200"]
    throughput = next(
        item
        for item in comparison.comparisons
        if item.metric == "throughput_per_s"
    )
    assert throughput.candidate_group == "h200"
    assert throughput.relative_improvement_pct > 0
    assert studies.load(comparison.id).id == comparison.id
