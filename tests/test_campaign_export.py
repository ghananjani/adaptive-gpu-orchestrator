import csv
import json

from adaptive_gpu_orchestrator.campaigns import (
    CampaignNCCLBaseline,
    CampaignProvenance,
    CampaignReport,
    CampaignRun,
    CampaignSpec,
    campaign_trial_rows,
    write_campaign_export,
)
from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.models import NCCLResult
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStore,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)


def make_campaign(tmp_path):
    store = OptimizationStore(tmp_path / "optimizations")
    first = TrialResult(
        config=ParallelismConfig(tensor_parallel=1),
        metrics=TrialMetrics(
            throughput_per_s=100.0,
            p99_latency_ms=120.0,
            extras={"ttft_ms": 80.0},
        ),
        score=100.0,
    )
    second = TrialResult(
        config=ParallelismConfig(tensor_parallel=2),
        metrics=TrialMetrics(
            throughput_per_s=140.0,
            p99_latency_ms=100.0,
            extras={"ttft_ms": 60.0},
        ),
        score=140.0,
    )
    result = OptimizationResult(
        id="opt1",
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[first, second],
        baseline=first,
        best=second,
    )
    store.save(result)

    report = CampaignReport(
        id="campaign1",
        spec=CampaignSpec(
            name="h100-vllm",
            group="h100",
            repeats=1,
            available_gpus=2,
            candidates=(first.config, second.config),
            objective=OptimizationObjective.THROUGHPUT,
            constraints=OptimizationConstraints(),
            adapter="vllm",
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
                "driver_version": "580.12",
                "cuda_version": "13.0",
                "gpus": [
                    {"name": "NVIDIA H100 80GB HBM3"},
                    {"name": "NVIDIA H100 80GB HBM3"},
                ],
            },
            package_versions={"vllm": "0.11.0"},
            cuda_toolkit_version="13.0",
            nccl_version="2.28.3",
            container_image="registry/llm:latest",
            container_image_digest="sha256:abc",
        ),
        runs=[
            CampaignRun(
                replicate=1,
                status="completed",
                optimization_id="opt1",
                best_config=second.config.label(),
            )
        ],
        nccl_baseline=CampaignNCCLBaseline(
            operation="all_reduce",
            source="saved-output",
            results=(
                NCCLResult(
                    operation="all_reduce",
                    size_bytes=1024,
                    time_us=2.0,
                    algbw_gbps=120.0,
                    busbw_gbps=110.0,
                ),
            ),
        ),
        manifest_sha256="manifest123",
    )
    return report, store


def test_campaign_trial_rows_flatten_trials_provenance_and_nccl(tmp_path):
    report, store = make_campaign(tmp_path)
    rows = campaign_trial_rows(report, optimization_store=store)

    assert len(rows) == 2
    assert rows[0]["campaign_group"] == "h100"
    assert rows[0]["gpu_count"] == 2
    assert rows[0]["gpu_models"] == "NVIDIA H100 80GB HBM3"
    assert rows[0]["cuda_toolkit_version"] == "13.0"
    assert rows[0]["nccl_version"] == "2.28.3"
    assert rows[0]["nccl_peak_busbw_gbps"] == 110.0
    assert rows[0]["extra_ttft_ms"] == 80.0
    assert rows[0]["selected_best"] is False
    assert rows[1]["selected_best"] is True


def test_campaign_export_writes_csv_and_jsonl(tmp_path):
    report, store = make_campaign(tmp_path)

    csv_path = write_campaign_export(
        report,
        optimization_store=store,
        output=tmp_path / "campaign.csv",
        format="csv",
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[1]["config_label"] == "tp2-pp1-dp1-b1-c1"
    assert rows[1]["extra_ttft_ms"] == "60.0"

    jsonl_path = write_campaign_export(
        report,
        optimization_store=store,
        output=tmp_path / "campaign.jsonl",
        format="jsonl",
    )
    lines = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    assert lines[0]["container_image_digest"] == "sha256:abc"
