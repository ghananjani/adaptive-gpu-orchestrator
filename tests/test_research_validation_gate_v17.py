"""Acceptance gate regression with disk-backed research experiments."""
import json

import pytest
from test_research_disk_e2e_v17 import build_saved_experiments

from adaptive_gpu_orchestrator.modeling.validation_gate import (
    ValidationPolicy,
    evaluate_validation_gate,
)
from adaptive_gpu_orchestrator.research_cli import main as generate
from adaptive_gpu_orchestrator.research_gate_cli import main as gate


def report_fixture(tmp_path):
    campaigns, optimizations, ids, hardware = build_saved_experiments(tmp_path)
    out = tmp_path / "reports"
    assert generate([
        *ids, "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--hardware-json", str(hardware), "--output-dir", str(out),
        "--iterations", "20", "--seed", "17",
    ]) == 0
    return out / "research-report.json"


def test_acceptance_gate_uses_actual_serialized_report(tmp_path, capsys):
    report = report_fixture(tmp_path)
    capsys.readouterr()
    result = gate([str(report), "--min-platforms", "3",
                   "--min-test-samples-per-platform", "6",
                   "--min-bootstrap-valid-fraction", "1"])
    assert result == 0, capsys.readouterr().err
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["passed"]
    assert outcome["evaluated_platforms"] == 3
    assert outcome["valid_bootstrap_fraction"] == 1


def test_failed_error_threshold_returns_distinct_status(tmp_path, capsys):
    report = report_fixture(tmp_path)
    capsys.readouterr()
    assert gate([str(report), "--max-conditioned-rmse", "0"]) == 3
    output = json.loads(capsys.readouterr().out)
    assert not output["passed"]
    assert any("conditioned RMSE exceeds limit" in s for s in output["failures"])


def test_synthetic_report_failure_for_small_folds_and_invalid_bootstrap(tmp_path):
    report = json.loads(report_fixture(tmp_path).read_text())
    policy = ValidationPolicy(min_platforms=4, min_test_samples_per_platform=7,
                              min_bootstrap_valid_fraction=0.95)
    report["summary"]["valid_iterations"] = 2
    result = evaluate_validation_gate(report, policy)
    assert not result["passed"]
    assert len(result["failures"]) >= 5


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_bad_bootstrap_requirement_is_rejected(value):
    with pytest.raises(ValueError):
        ValidationPolicy(min_bootstrap_valid_fraction=value)


def test_rejects_malformed_report_and_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    assert gate([str(missing)]) == 2
    assert "research gate failed" in capsys.readouterr().err
    with pytest.raises(ValueError, match="invalid research report kind"):
        evaluate_validation_gate({"kind": "something-else"}, ValidationPolicy())
