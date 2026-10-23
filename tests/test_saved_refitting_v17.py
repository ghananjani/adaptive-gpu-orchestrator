import json
from types import SimpleNamespace

import pytest

from adaptive_gpu_orchestrator.modeling import campaign_evidence, design_audit, saved_refitting
from adaptive_gpu_orchestrator.modeling.hardware import HardwareFeatures

TAGS = {
    "model": "test-model", "workload": "benchmark",
    "precision": "bf16", "input_tokens": "3000",
    "output_tokens": "128", "benchmark_protocol": "v1",
}


def fixtures(monkeypatch):
    campaigns, rows, hardware = [], {}, {}
    for idx, platform in enumerate(("GPU-A", "GPU-B", "GPU-C")):
        hardware[platform] = HardwareFeatures(
            gpu_count=8, memory_gib_per_gpu=64 + 16 * idx,
            memory_bandwidth_gbps_per_gpu=1200 + 200 * idx,
            compute_tflops_per_gpu=40 + idx * 10,
            interconnect_bandwidth_gbps=300 + idx * 50,
        )
        for camp in range(3):
            cid = f"{platform}-{camp}"
            campaigns.append(SimpleNamespace(id=cid, spec=SimpleNamespace(tags=dict(TAGS))))
            rows[cid] = [
                {
                    "gpu_models": platform, "gpu_count": 8, "exit_code": 0,
                    "cuda_runtime_version": "12.4", "cuda_toolkit_version": "12.4",
                    "nccl_version": "2.22", "container_image_digest": "sha256:fixture",
                    "tensor_parallel": tp, "pipeline_parallel": 1,
                    "data_parallel": 1, "batch_size": 1, "concurrency": 1,
                    "throughput_per_s": float(85 + idx * 12 + camp + tp * 5),
                    "replicate": 1, "trial_index": trial_index,
                    "optimization_id": f"optimization-{cid}",
                }
                for trial_index, tp in enumerate((1, 2, 4), start=1)
            ]
    getter = lambda campaign, *, optimization_store: rows[campaign.id]
    monkeypatch.setattr(saved_refitting, "campaign_trial_rows", getter)
    monkeypatch.setattr(campaign_evidence, "campaign_trial_rows", getter)
    monkeypatch.setattr(design_audit, "campaign_trial_rows", getter)
    return campaigns, rows, hardware


def test_end_to_end_reproducible_export(monkeypatch, tmp_path):
    campaigns, _, hardware = fixtures(monkeypatch)
    kwargs = {
        "optimization_store": object(), "measured_hardware": hardware,
        "iterations": 20, "seed": 23,
    }
    a = saved_refitting.evaluate_saved_refitting(campaigns, **kwargs)
    b = saved_refitting.evaluate_saved_refitting(reversed(campaigns), **kwargs)
    assert a.to_dict() == b.to_dict()
    assert a.refitting.campaign_count == 9
    assert a.refitting.test_samples == 27
    assert a.refitting.valid_iterations == 20
    assert a.design_audit.independent_campaign_replication
    p = saved_refitting.write_saved_refitting_report(a, tmp_path / "research.json")
    assert json.loads(p.read_text())["refitting"]["seed"] == 23


def test_evidence_reuse_blocked(monkeypatch):
    campaigns, rows, hardware = fixtures(monkeypatch)
    rows[campaigns[1].id][0]["optimization_id"] = "optimization-GPU-A-0"
    with pytest.raises(ValueError, match="shared optimization"):
        saved_refitting.evaluate_saved_refitting(
            campaigns, optimization_store=object(),
            measured_hardware=hardware, iterations=20,
        )


def test_protocol_mismatch_blocked(monkeypatch):
    campaigns, _, hardware = fixtures(monkeypatch)
    campaigns[0].spec.tags["precision"] = "fp8"
    with pytest.raises(ValueError, match="tags mismatch"):
        saved_refitting.evaluate_saved_refitting(
            campaigns, optimization_store=object(),
            measured_hardware=hardware, iterations=20,
        )


def test_missing_campaign_replication_blocked(monkeypatch):
    campaigns, _, hardware = fixtures(monkeypatch)
    only_one = [c for c in campaigns if c.id.endswith("-0")]
    with pytest.raises(ValueError, match="two separate campaigns"):
        saved_refitting.evaluate_saved_refitting(
            only_one, optimization_store=object(),
            measured_hardware=hardware, iterations=20,
        )


def test_exclusion_accounting(monkeypatch):
    campaigns, rows, hardware = fixtures(monkeypatch)
    rows[campaigns[0].id][0]["exit_code"] = 1
    rows[campaigns[0].id][1]["throughput_per_s"] = float("nan")
    report = saved_refitting.evaluate_saved_refitting(
        campaigns, optimization_store=object(),
        measured_hardware=hardware, iterations=20,
    )
    assert report.comparison.excluded_failed == 1
    assert report.comparison.excluded_missing == 1
    assert report.refitting.test_samples == 25
