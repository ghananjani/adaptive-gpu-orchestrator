from types import SimpleNamespace

import pytest

from adaptive_gpu_orchestrator.modeling import design_audit


def fixtures(monkeypatch, *, campaigns_per_platform=1, shared=False):
    campaigns = []
    rows = {}
    for p in ("H100", "H200", "L40S"):
        for index in range(campaigns_per_platform):
            cid = f"{p}-{index}"
            campaigns.append(SimpleNamespace(id=cid))
            rows[cid] = [{
                "gpu_models": p,
                "optimization_id": "reused" if shared else cid,
                "replicate": 1,
                "cuda_runtime_version": "12.4",
                "cuda_toolkit_version": "12.4",
                "nccl_version": "2.22",
                "container_image_digest": "sha256:test",
            }]
    monkeypatch.setattr(
        design_audit,
        "campaign_trial_rows",
        lambda campaign, *, optimization_store: rows[campaign.id],
    )
    return campaigns, rows


def test_single_campaign_per_platform_detects_confounding(monkeypatch):
    campaigns, _ = fixtures(monkeypatch)
    report = design_audit.audit_campaign_design(campaigns, optimization_store=object())
    assert report.platform_campaign_confounded
    assert not report.independent_campaign_replication
    assert report.platform_count == 3
    assert report.to_dict()["schema_version"] == 1


def test_independent_campaigns_reported_distinctly_from_trial_repeats(monkeypatch):
    campaigns, _ = fixtures(monkeypatch, campaigns_per_platform=2)
    forward = design_audit.audit_campaign_design(campaigns, optimization_store=object())
    backward = design_audit.audit_campaign_design(reversed(campaigns), optimization_store=object())
    assert forward.to_dict() == backward.to_dict()
    assert forward.independent_campaign_replication
    assert not forward.platform_campaign_confounded


def test_reused_optimization_evidence_is_flagged(monkeypatch):
    campaigns, _ = fixtures(monkeypatch, shared=True)
    report = design_audit.audit_campaign_design(campaigns, optimization_store=object())
    assert report.shared_optimization_ids == ("reused",)
    assert any("reuse/leakage" in message for message in report.warnings)


def test_missing_software_provenance_is_visible(monkeypatch):
    campaigns, rows = fixtures(monkeypatch)
    rows["H100-0"][0]["nccl_version"] = None
    report = design_audit.audit_campaign_design(campaigns, optimization_store=object())
    assert report.missing_provenance["H100-0"] == ("nccl_version",)


def test_duplicate_campaign_rejected(monkeypatch):
    campaigns, _ = fixtures(monkeypatch)
    with pytest.raises(ValueError, match="unique IDs"):
        design_audit.audit_campaign_design([campaigns[0], campaigns[0]], optimization_store=object())


def test_ambiguous_platform_rejected(monkeypatch):
    campaigns, rows = fixtures(monkeypatch)
    rows["H100-0"].append({**rows["H100-0"][0], "gpu_models": "H200"})
    with pytest.raises(ValueError, match="ambiguous hardware"):
        design_audit.audit_campaign_design(campaigns, optimization_store=object())
