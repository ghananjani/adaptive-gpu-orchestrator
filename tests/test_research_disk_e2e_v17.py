"""True disk-backed research pipeline regression: no mocked stores or exports."""
from __future__ import annotations

import hashlib
import json

from adaptive_gpu_orchestrator.campaigns import (
    CampaignProvenance,
    CampaignReport,
    CampaignRun,
    CampaignSpec,
    CampaignStore,
    campaign_trial_rows,
)
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
from adaptive_gpu_orchestrator.research_cli import main

TAGS = {
    "model": "synthetic-fixture", "workload": "synthetic-prompt",
    "precision": "bf16", "input_tokens": "3000",
    "output_tokens": "128", "benchmark_protocol": "synthetic-v1",
}


def build_saved_experiments(tmp_path):
    campaign_store = CampaignStore(tmp_path / "campaigns")
    optimization_store = OptimizationStore(tmp_path / "optimizations")
    hardware = {}
    campaign_ids = []
    for platform_number, platform in enumerate(("Synthetic-A", "Synthetic-B", "Synthetic-C")):
        hardware[platform] = {
            "gpu_count": 1,
            "memory_gib_per_gpu": float(64 + 16 * platform_number),
            "memory_bandwidth_gbps_per_gpu": float(1000 + 500 * platform_number),
            "compute_tflops_per_gpu": float(40 + 20 * platform_number),
            "interconnect_bandwidth_gbps": float(100 + 50 * platform_number),
        }
        for campaign_number in range(2):
            campaign_id = f"{platform}-{campaign_number}"
            optimization_id = f"optimization-{campaign_id}"
            configs = [
                ParallelismConfig(
                    tensor_parallel=1, pipeline_parallel=1, data_parallel=1,
                    batch_size=1, concurrency=concurrency,
                )
                for concurrency in (1, 2, 4)
            ]
            trials = [
                TrialResult(
                    config=config,
                    metrics=TrialMetrics(
                        throughput_per_s=float(
                            80 + 25 * platform_number + 5 * campaign_number + 3 * position
                        ),
                        exit_code=0,
                    ),
                    feasible=True,
                )
                for position, config in enumerate(configs)
            ]
            optimization_store.save(OptimizationResult(
                id=optimization_id,
                objective=OptimizationObjective.THROUGHPUT,
                constraints=OptimizationConstraints(),
                trials=trials,
                best=trials[-1],
            ))
            campaign_store.save(CampaignReport(
                id=campaign_id,
                spec=CampaignSpec(
                    name=campaign_id, group="synthetic-integration",
                    repeats=1, available_gpus=1,
                    candidates=tuple(configs),
                    objective=OptimizationObjective.THROUGHPUT,
                    constraints=OptimizationConstraints(),
                    adapter="synthetic", benchmark_command=("echo", "synthetic"),
                    tags=dict(TAGS),
                ),
                provenance=CampaignProvenance(
                    runtime=RuntimeProvenance(
                        python_version="3.10", implementation="CPython",
                        platform="synthetic", machine="synthetic",
                    ),
                    cluster_snapshot={
                        "gpus": [{"name": platform}],
                        "driver_version": "synthetic",
                        "cuda_version": "12.4",
                    },
                    package_versions={},
                    cuda_toolkit_version="12.4", nccl_version="2.22",
                    container_image_digest="sha256:synthetic-test",
                ),
                runs=[CampaignRun(
                    replicate=1, status="completed", optimization_id=optimization_id
                )],
            ))
            campaign_ids.append(campaign_id)
    hardware_path = tmp_path / "hardware.json"
    hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
    return campaign_store, optimization_store, campaign_ids, hardware_path


def test_disk_backed_campaign_store_export_research_cli_end_to_end(tmp_path, capsys):
    campaigns, optimizations, ids, hardware_path = build_saved_experiments(tmp_path)
    loaded = campaigns.load(ids[0])
    assert loaded.id == ids[0]
    assert optimizations.load(f"optimization-{ids[0]}").id == f"optimization-{ids[0]}"
    actual_rows = campaign_trial_rows(loaded, optimization_store=optimizations)
    assert len(actual_rows) == 3
    assert actual_rows[0]["gpu_models"] == "Synthetic-A"
    assert actual_rows[0]["gpu_count"] == 1
    assert actual_rows[0]["optimization_id"] == f"optimization-{ids[0]}"

    out = tmp_path / "reports"
    argv = [
        *ids, "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--hardware-json", str(hardware_path),
        "--output-dir", str(out), "--iterations", "20", "--seed", "17",
    ]
    assert main(argv) == 0, capsys.readouterr().err
    data = json.loads((out / "research-report.json").read_text())
    assert data["kind"] == "gpu-hardware-research-report"
    assert data["summary"]["campaign_count"] == 6
    assert data["summary"]["trial_count"] == 18
    assert data["summary"]["platform_count"] == 3
    assert data["summary"]["valid_iterations"] == 20
    assert data["source"]["design_audit"]["independent_campaign_replication"]
    assert len(data["source"]["comparison"]["evaluation"]["folds"]) == 3
    assert (out / "research-report.html").is_file()
    assert (out / "research-report.md").is_file()
    assert len((out / "heldout-folds.csv").read_text().splitlines()) == 4
    digest = hashlib.sha256((out / "research-report.json").read_bytes()).hexdigest()
    assert (out / "research-report.sha256").read_text().startswith(digest)
    first = {path.name: path.read_bytes() for path in out.iterdir()}
    assert main(argv) == 0, capsys.readouterr().err
    assert first == {path.name: path.read_bytes() for path in out.iterdir()}


def test_disk_backed_research_pipeline_rejects_protocol_mismatch(tmp_path, capsys):
    campaigns, optimizations, ids, hardware_path = build_saved_experiments(tmp_path)
    altered = campaigns.load(ids[-1])
    altered.spec.tags["precision"] = "int4"
    campaigns.save(altered)
    out = tmp_path / "rejected"
    assert main([
        *ids, "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--hardware-json", str(hardware_path), "--output-dir", str(out),
        "--iterations", "20",
    ]) == 2
    assert "tags mismatch" in capsys.readouterr().err
    assert not out.exists()
