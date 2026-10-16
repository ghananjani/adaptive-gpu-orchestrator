from adaptive_gpu_orchestrator.campaigns import (
    CampaignProvenance,
    CampaignReport,
    CampaignRun,
    CampaignSpec,
    CampaignStore,
)
from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    ParallelismConfig,
)


def test_campaign_store_round_trip(tmp_path):
    report = CampaignReport(
        id="campaign1",
        spec=CampaignSpec(
            name="h100 benchmark",
            group="h100",
            repeats=2,
            available_gpus=1,
            candidates=(ParallelismConfig(),),
            objective=OptimizationObjective.THROUGHPUT,
            constraints=OptimizationConstraints(),
            adapter="generic",
            benchmark_command=("python", "bench.py"),
            tags={"hardware": "h100"},
        ),
        provenance=CampaignProvenance(
            runtime=RuntimeProvenance(
                python_version="3.12.0",
                implementation="cpython",
                platform="Linux-test",
                machine="x86_64",
            ),
            cluster_snapshot={"hostname": "node"},
            package_versions={"torch": "2.9.0"},
        ),
        runs=[
            CampaignRun(
                replicate=1,
                status="completed",
                optimization_id="opt1",
                best_config="tp1-pp1-dp1-b1-c1",
            ),
            CampaignRun(
                replicate=2,
                status="error",
                error="benchmark failed",
            ),
        ],
        study_id="study1",
        findings=["1/2 completed"],
    )

    store = CampaignStore(tmp_path)
    path = store.save(report)
    loaded = store.load("campaign1")

    assert path.exists()
    assert loaded.id == report.id
    assert loaded.spec == report.spec
    assert loaded.provenance == report.provenance
    assert loaded.runs == report.runs
    assert loaded.study_id == "study1"
    assert loaded.successful_replicates == 1
