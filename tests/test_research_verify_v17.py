"""End-to-end verifier regression with genuine persisted stores, no GPU runs."""
import json

from test_research_disk_e2e_v17 import build_saved_experiments

from adaptive_gpu_orchestrator.research_cli import main as generate
from adaptive_gpu_orchestrator.research_verify import main as verify


def setup_report(tmp_path):
    campaigns, optimizations, ids, hardware = build_saved_experiments(tmp_path)
    out = tmp_path / "reports"
    assert generate([
        *ids, "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--hardware-json", str(hardware), "--output-dir", str(out),
        "--iterations", "20", "--seed", "17",
    ]) == 0
    command = [
        str(out / "research-report.json"),
        "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--checksum", str(out / "research-report.sha256"),
    ]
    return campaigns, optimizations, ids, out, command


def test_verification_and_checksum_round_trip(tmp_path, capsys):
    _, _, ids, _, command = setup_report(tmp_path)
    capsys.readouterr()
    assert verify(command) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["campaign_count"] == len(ids)
    assert result["report_checksum_verified"] is True


def test_rejects_changed_campaign_even_if_report_checksum_valid(tmp_path, capsys):
    campaigns, _, ids, _, command = setup_report(tmp_path)
    altered = campaigns.load(ids[0])
    altered.findings.append("changed source")
    campaigns.save(altered)
    assert verify(command) == 2
    assert "fingerprint mismatch" in capsys.readouterr().err


def test_rejects_changed_optimization(tmp_path, capsys):
    _, optimizations, ids, _, command = setup_report(tmp_path)
    altered = optimizations.load("optimization-" + ids[0])
    altered.trials[0].score = 1.0
    optimizations.save(altered)
    assert verify(command) == 2
    assert "fingerprint mismatch" in capsys.readouterr().err


def test_rejects_report_corruption_with_checksum(tmp_path, capsys):
    _, _, _, out, command = setup_report(tmp_path)
    report = out / "research-report.json"
    data = json.loads(report.read_text())
    data["summary"]["trial_count"] = 999
    report.write_text(json.dumps(data), encoding="utf-8")
    assert verify(command) == 2
    assert "checksum mismatch" in capsys.readouterr().err


def test_rejects_missing_manifest(tmp_path, capsys):
    _, _, _, out, command = setup_report(tmp_path)
    report = out / "research-report.json"
    data = json.loads(report.read_text())
    data.pop("evidence_manifest")
    report.write_text(json.dumps(data), encoding="utf-8")
    assert verify(command) == 2
    assert "lacks source evidence manifest" in capsys.readouterr().err
