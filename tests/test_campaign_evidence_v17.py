import json
from types import SimpleNamespace

import pytest

from adaptive_gpu_orchestrator.modeling import campaign_evidence
from adaptive_gpu_orchestrator.modeling.hardware import HardwareFeatures

TAGS = {
    "model": "test-model", "workload": "prefill-decode",
    "precision": "bf16", "input_tokens": "3072",
    "output_tokens": "128", "benchmark_protocol": "fixture-v1",
}


def setup(monkeypatch, *, software=False, mixed_tags=False):
    campaigns = []
    rows_by_id = {}
    hardware = {}
    for index, name in enumerate(("GPU-A", "GPU-B", "GPU-C")):
        tags = dict(TAGS)
        if mixed_tags and index == 1:
            tags["precision"] = "fp8"
        campaign = SimpleNamespace(
            id=f"campaign-{index}", spec=SimpleNamespace(tags=tags),
        )
        campaigns.append(campaign)
        hardware[name] = HardwareFeatures(
            gpu_count=8,
            memory_gib_per_gpu=64 + index * 16,
            memory_bandwidth_gbps_per_gpu=1200 + index * 300,
            compute_tflops_per_gpu=50 + index * 20,
            interconnect_bandwidth_gbps=300 + index * 100,
        )
        rows_by_id[campaign.id] = [
            {
                "gpu_models": name, "gpu_count": 8, "exit_code": 0,
                "cuda_runtime_version": "12.4" if not software or index != 2 else "12.5",
                "cuda_toolkit_version": "12.4", "nccl_version": "2.22",
                "container_image_digest": "sha256:fixture",
                "tensor_parallel": tp, "pipeline_parallel": 1,
                "data_parallel": 1, "batch_size": 1, "concurrency": 1,
                "throughput_per_s": 100 + 10 * index + 20 * (tp.bit_length() - 1),
                "replicate": 1, "trial_index": trial_index,
            }
            for trial_index, tp in enumerate((1, 2, 4, 8), start=1)
        ]
    monkeypatch.setattr(
        campaign_evidence, "campaign_trial_rows",
        lambda campaign, *, optimization_store: rows_by_id[campaign.id],
    )
    return campaigns, hardware, rows_by_id


def test_end_to_end_campaign_export_and_deterministic_digest(monkeypatch, tmp_path):
    campaigns, hardware, _ = setup(monkeypatch)
    result = campaign_evidence.compare_saved_campaigns(
        campaigns, optimization_store=object(), measured_hardware=hardware,
    )
    reversed_result = campaign_evidence.compare_saved_campaigns(
        reversed(campaigns), optimization_store=object(), measured_hardware=hardware,
    )
    assert result.to_dict() == reversed_result.to_dict()
    assert result.trial_count == 12
    assert result.evaluation.test_samples == 12
    assert len(result.source_sha256) == 64
    json_file, csv_file = campaign_evidence.write_campaign_comparison(
        result, json_path=tmp_path / "comparison.json", csv_path=tmp_path / "folds.csv",
    )
    assert json.loads(json_file.read_text())["source_sha256"] == result.source_sha256
    assert csv_file.read_text().count("\n") == 4


def test_tag_mismatch_is_blocked(monkeypatch):
    campaigns, hardware, _ = setup(monkeypatch, mixed_tags=True)
    with pytest.raises(ValueError, match="tags mismatch"):
        campaign_evidence.compare_saved_campaigns(
            campaigns, optimization_store=object(), measured_hardware=hardware,
        )


def test_mismatched_software_requires_override(monkeypatch):
    campaigns, hardware, _ = setup(monkeypatch, software=True)
    with pytest.raises(ValueError, match="software provenance"):
        campaign_evidence.compare_saved_campaigns(
            campaigns, optimization_store=object(), measured_hardware=hardware,
        )
    result = campaign_evidence.compare_saved_campaigns(
        campaigns, optimization_store=object(), measured_hardware=hardware,
        allow_software_mismatch=True,
    )
    assert result.warnings


def test_gpu_count_and_hardware_coverage_are_strict(monkeypatch):
    campaigns, hardware, _ = setup(monkeypatch)
    hardware.pop("GPU-B")
    with pytest.raises(ValueError, match="missing measured hardware"):
        campaign_evidence.compare_saved_campaigns(
            campaigns, optimization_store=object(), measured_hardware=hardware,
        )


def test_failed_and_missing_values_are_excluded(monkeypatch):
    campaigns, hardware, rows = setup(monkeypatch)
    rows["campaign-0"][0]["exit_code"] = 1
    rows["campaign-0"][1]["throughput_per_s"] = None
    result = campaign_evidence.compare_saved_campaigns(
        campaigns, optimization_store=object(), measured_hardware=hardware,
    )
    assert result.excluded_failed == 1
    assert result.excluded_missing == 1
    assert result.trial_count == 10


def test_no_missing_provenance_accepted(monkeypatch):
    campaigns, hardware, _ = setup(monkeypatch)
    campaigns[0].spec.tags.pop("precision")
    with pytest.raises(ValueError, match="missing required tag"):
        campaign_evidence.compare_saved_campaigns(
            campaigns, optimization_store=object(), measured_hardware=hardware,
        )
