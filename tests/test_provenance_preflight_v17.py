"""Regression guards for the strict saved research report metadata preflight."""
import pytest
from test_saved_refitting_v17 import fixtures

from adaptive_gpu_orchestrator.modeling import saved_refitting


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nccl_version", None, "missing software provenance: nccl_version"),
        ("container_image_digest", "", "missing software provenance: container_image_digest"),
        ("gpu_count", None, "missing valid GPU count"),
        ("gpu_count", 7, "GPU count mismatch"),
        ("optimization_id", "", "missing optimization ID"),
        ("trial_index", None, "duplicated or missing trial identity"),
    ],
)
def test_incomplete_metadata_fails_before_analysis(monkeypatch, field, value, message):
    campaigns, rows, hardware = fixtures(monkeypatch)
    rows[campaigns[0].id][0][field] = value
    with pytest.raises(ValueError, match=message):
        saved_refitting.evaluate_saved_refitting(
            campaigns, optimization_store=object(),
            measured_hardware=hardware, iterations=20,
        )


def test_duplicate_trial_identity_rejected(monkeypatch):
    campaigns, rows, hardware = fixtures(monkeypatch)
    rows[campaigns[0].id][1]["trial_index"] = rows[campaigns[0].id][0]["trial_index"]
    with pytest.raises(ValueError, match="duplicated or missing trial identity"):
        saved_refitting.evaluate_saved_refitting(
            campaigns, optimization_store=object(),
            measured_hardware=hardware, iterations=20,
        )


def test_mixed_hardware_inside_campaign_rejected(monkeypatch):
    campaigns, rows, hardware = fixtures(monkeypatch)
    rows[campaigns[0].id][0]["gpu_models"] = "GPU-B"
    with pytest.raises(ValueError, match="mixes hardware platforms"):
        saved_refitting.evaluate_saved_refitting(
            campaigns, optimization_store=object(),
            measured_hardware=hardware, iterations=20,
        )


def test_complete_saved_campaigns_still_pass(monkeypatch):
    campaigns, _, hardware = fixtures(monkeypatch)
    report = saved_refitting.evaluate_saved_refitting(
        campaigns, optimization_store=object(),
        measured_hardware=hardware, iterations=20,
    )
    assert report.refitting.test_samples == 27
