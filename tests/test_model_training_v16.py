import pytest

from adaptive_gpu_orchestrator.campaigns import (
    CampaignProvenance,
    CampaignReport,
    CampaignRun,
    CampaignSpec,
)
from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.modeling import train_performance_model
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStore,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)


def make_campaign(
    *,
    identifier,
    gpu_name,
    optimization_store,
    throughput_offset=0.0,
):
    configs = [
        ParallelismConfig(tensor_parallel=1),
        ParallelismConfig(tensor_parallel=2),
        ParallelismConfig(tensor_parallel=4),
        ParallelismConfig(tensor_parallel=8),
    ]
    trials = [
        TrialResult(
            config=config,
            metrics=TrialMetrics(
                throughput_per_s=100.0 + throughput_offset + index * 25.0,
                p99_latency_ms=200.0 - index * 20.0,
                energy_joules=1000.0 - index * 50.0,
                memory_pressure_pct=60.0 + index * 5.0,
            ),
            score=100.0 + throughput_offset + index * 25.0,
        )
        for index, config in enumerate(configs)
    ]
    result = OptimizationResult(
        id=f"opt-{identifier}",
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=trials,
        baseline=trials[0],
        best=trials[-1],
    )
    optimization_store.save(result)

    return CampaignReport(
        id=identifier,
        spec=CampaignSpec(
            name=identifier,
            group=identifier,
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
                    {"name": gpu_name}
                    for _ in range(8)
                ],
            },
            package_versions={"torch": "2.9.0"},
            cuda_toolkit_version="13.0",
            nccl_version="2.28.3",
            container_image_digest="sha256:abc",
        ),
        runs=[
            CampaignRun(
                replicate=1,
                status="completed",
                optimization_id=result.id,
                best_config=trials[-1].config.label(),
            )
        ],
        manifest_sha256=f"manifest-{identifier}",
    )


def test_train_performance_model_from_campaign_trials(tmp_path):
    store = OptimizationStore(tmp_path / "optimizations")
    campaign = make_campaign(
        identifier="h100",
        gpu_name="NVIDIA H100",
        optimization_store=store,
    )

    report = train_performance_model(
        [campaign],
        optimization_store=store,
        metrics=("throughput_per_s", "p99_latency_ms"),
        ridge_lambda=0.1,
        min_samples=3,
    )

    assert report.campaign_ids == ("h100",)
    assert len(report.hardware_signatures) == 1
    assert report.metrics["throughput_per_s"].sample_count == 4
    assert report.metrics["p99_latency_ms"].distinct_config_count == 4
    assert len(report.observed_configs) == 4


def test_mixed_hardware_requires_explicit_override(tmp_path):
    store = OptimizationStore(tmp_path / "optimizations")
    h100 = make_campaign(
        identifier="h100",
        gpu_name="NVIDIA H100",
        optimization_store=store,
    )
    h200 = make_campaign(
        identifier="h200",
        gpu_name="NVIDIA H200",
        optimization_store=store,
        throughput_offset=20.0,
    )

    with pytest.raises(ValueError, match="multiple hardware/software"):
        train_performance_model(
            [h100, h200],
            optimization_store=store,
            metrics=("throughput_per_s",),
        )

    report = train_performance_model(
        [h100, h200],
        optimization_store=store,
        metrics=("throughput_per_s",),
        allow_mixed_hardware=True,
    )
    assert len(report.hardware_signatures) == 2
    assert any(
        "mixed-hardware override" in finding
        for finding in report.findings
    )
