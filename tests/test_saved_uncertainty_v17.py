from types import SimpleNamespace

import pytest

from adaptive_gpu_orchestrator.modeling import saved_uncertainty
from adaptive_gpu_orchestrator.modeling.hardware import HardwareFeatures

TAGS = {
    "model": "fixture", "workload": "text-generation", "precision": "bf16",
    "input_tokens": "3000", "output_tokens": "128",
    "benchmark_protocol": "stable-test",
}


def setup(monkeypatch, *, campaigns_per_platform=2):
    campaigns = []
    rows = {}
    hardware = {}
    for index, name in enumerate(("Platform-A", "Platform-B", "Platform-C")):
        hardware[name] = HardwareFeatures(
            gpu_count=8, memory_gib_per_gpu=64 + 16 * index,
            memory_bandwidth_gbps_per_gpu=1200 + 400 * index,
            compute_tflops_per_gpu=40 + 15 * index,
            interconnect_bandwidth_gbps=300 + 100 * index,
        )
        for repeat in range(campaigns_per_platform):
            campaign = SimpleNamespace(
                id=f"{name}-{repeat}", spec=SimpleNamespace(tags=dict(TAGS))
            )
            campaigns.append(campaign)
            rows[campaign.id] = [
                {
                    "gpu_models": name, "gpu_count": 8, "exit_code": 0,
                    "cuda_runtime_version": "12.4", "cuda_toolkit_version": "12.4",
                    "nccl_version": "2.22", "container_image_digest": "sha256:fixture",
                    "tensor_parallel": tp, "pipeline_parallel": 1,
                    "data_parallel": 1, "batch_size": 1, "concurrency": 1,
                    "throughput_per_s": 95.0 + index * 15 + tp * 4 + repeat,
                    "replicate": 1, "trial_index": idx,
                }
                for idx, tp in enumerate((1, 2, 4, 8), start=1)
            ]
    monkeypatch.setattr(
        saved_uncertainty, "campaign_trial_rows",
        lambda campaign, *, optimization_store: rows[campaign.id],
    )
    from adaptive_gpu_orchestrator.modeling import campaign_evidence
    monkeypatch.setattr(
        campaign_evidence, "campaign_trial_rows",
        lambda campaign, *, optimization_store: rows[campaign.id],
    )
    return campaigns, hardware, rows


def test_saved_evidence_produces_reproducible_residuals(monkeypatch):
    campaigns, hardware, _ = setup(monkeypatch)
    kwargs = {
        "optimization_store": object(), "measured_hardware": hardware,
        "iterations": 100, "seed": 19,
    }
    first = saved_uncertainty.evaluate_saved_campaign_uncertainty(campaigns, **kwargs)
    second = saved_uncertainty.evaluate_saved_campaign_uncertainty(
        reversed(campaigns), **kwargs
    )
    assert first.to_dict() == second.to_dict()
    assert first.bootstrap.sample_count == 24
    assert first.bootstrap.campaign_count == 6
    assert first.bootstrap.valid_iterations == 100
    assert first.comparison.evaluation.test_samples == 24
    assert len(first.residuals) == 24


def test_requires_independent_campaign_replication(monkeypatch):
    campaigns, hardware, _ = setup(monkeypatch, campaigns_per_platform=1)
    with pytest.raises(ValueError, match="independent campaigns"):
        saved_uncertainty.evaluate_saved_campaign_uncertainty(
            campaigns, optimization_store=object(), measured_hardware=hardware,
            iterations=100,
        )


def test_provenance_mismatch_still_rejected(monkeypatch):
    campaigns, hardware, _ = setup(monkeypatch)
    campaigns[0].spec.tags["precision"] = "fp8"
    with pytest.raises(ValueError, match="tags mismatch"):
        saved_uncertainty.evaluate_saved_campaign_uncertainty(
            campaigns, optimization_store=object(), measured_hardware=hardware,
            iterations=100,
        )


def test_excluded_trials_match_saved_comparison(monkeypatch):
    campaigns, hardware, rows = setup(monkeypatch)
    rows[campaigns[0].id][0]["exit_code"] = 1
    rows[campaigns[0].id][1]["throughput_per_s"] = float("nan")
    report = saved_uncertainty.evaluate_saved_campaign_uncertainty(
        campaigns, optimization_store=object(), measured_hardware=hardware,
        iterations=100,
    )
    assert report.comparison.excluded_failed == 1
    assert report.comparison.excluded_missing == 1
    assert len(report.residuals) == 22
