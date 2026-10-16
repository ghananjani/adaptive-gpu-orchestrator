from adaptive_gpu_orchestrator.campaigns import (
    CampaignProvenance,
    CampaignReport,
    CampaignRun,
    CampaignSpec,
    CampaignStore,
)
from adaptive_gpu_orchestrator.cli import main
from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStore,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)


def seed_campaign(tmp_path):
    optimization_store = OptimizationStore(tmp_path / "optimizations")
    configs = [
        ParallelismConfig(tensor_parallel=1),
        ParallelismConfig(tensor_parallel=2),
        ParallelismConfig(tensor_parallel=4),
    ]
    trials = [
        TrialResult(
            config=config,
            metrics=TrialMetrics(
                throughput_per_s=value,
                p99_latency_ms=latency,
            ),
            score=value,
        )
        for config, value, latency in zip(
            configs,
            (100.0, 130.0, 165.0),
            (180.0, 150.0, 125.0),
        )
    ]
    optimization = OptimizationResult(
        id="opt-cli-model",
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=trials,
        baseline=trials[0],
        best=trials[-1],
    )
    optimization_store.save(optimization)

    campaign = CampaignReport(
        id="campaign-cli-model",
        spec=CampaignSpec(
            name="cli-model",
            group="h100",
            repeats=1,
            available_gpus=8,
            candidates=tuple(configs),
            objective=OptimizationObjective.THROUGHPUT,
            constraints=OptimizationConstraints(),
            adapter="generic",
            benchmark_command=("python", "bench.py"),
        ),
        provenance=CampaignProvenance(
            runtime=RuntimeProvenance(
                python_version="3.12.0",
                implementation="cpython",
                platform="Linux",
                machine="x86_64",
            ),
            cluster_snapshot={
                "hostname": "node",
                "driver_version": "580.12",
                "cuda_version": "13.0",
                "gpus": [
                    {"name": "NVIDIA H100"}
                    for _ in range(8)
                ],
            },
            package_versions={},
            cuda_toolkit_version="13.0",
            nccl_version="2.28.3",
        ),
        runs=[
            CampaignRun(
                replicate=1,
                status="completed",
                optimization_id=optimization.id,
                best_config=configs[-1].label(),
            )
        ],
    )
    campaign_store = CampaignStore(tmp_path / "campaigns")
    campaign_store.save(campaign)
    return campaign, campaign_store, optimization_store


def test_model_cli_train_then_recommend(tmp_path, capsys):
    campaign, campaign_store, optimization_store = seed_campaign(tmp_path)
    model_store = tmp_path / "models"

    code = main(
        [
            "model",
            "train",
            campaign.id,
            "--campaign-store",
            str(campaign_store.root),
            "--optimization-store",
            str(optimization_store.root),
            "--store-dir",
            str(model_store),
            "--metric",
            "throughput_per_s",
            "--ridge-lambda",
            "0.1",
        ]
    )
    output = capsys.readouterr()

    assert code == 0
    assert "Performance model:" in output.out
    paths = list(model_store.glob("*.json"))
    assert len(paths) == 1
    model_id = paths[0].stem

    recommend_code = main(
        [
            "model",
            "recommend",
            model_id,
            "--store-dir",
            str(model_store),
            "--metric",
            "throughput_per_s",
            "--gpus",
            "8",
            "--tp",
            "1,2,4,8",
            "--pp",
            "1",
            "--dp",
            "1",
            "--batch",
            "1",
            "--concurrency",
            "1",
            "--exploration-weight",
            "0",
            "--top-k",
            "1",
        ]
    )
    recommendation = capsys.readouterr()

    assert recommend_code == 0
    assert "Recommendations:" in recommendation.out
    assert "tp8-pp1-dp1-b1-c1" in recommendation.out
