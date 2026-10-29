"""Source evidence manifest checks using actual serialized campaign stores."""
import hashlib
import json

from test_research_disk_e2e_v17 import build_saved_experiments

from adaptive_gpu_orchestrator.modeling.evidence_fingerprint import fingerprint_saved_evidence
from adaptive_gpu_orchestrator.research_cli import main


def test_manifest_is_stable_and_detects_source_mutation(tmp_path):
    campaigns, optimizations, ids, _ = build_saved_experiments(tmp_path)
    reports = [campaigns.load(identifier) for identifier in ids]
    first = fingerprint_saved_evidence(reports, optimization_store=optimizations)
    second = fingerprint_saved_evidence(reversed(reports), optimization_store=optimizations)
    assert first == second
    assert len(first["sources"]) == 6
    assert len(first["manifest_sha256"]) == 64
    changed = campaigns.load(ids[0])
    changed.findings.append("additional metadata")
    campaigns.save(changed)
    after = fingerprint_saved_evidence(
        [campaigns.load(identifier) for identifier in ids],
        optimization_store=optimizations,
    )
    assert first["manifest_sha256"] != after["manifest_sha256"]
    optimization = optimizations.load("optimization-" + ids[0])
    optimization.trials[0].score = 1.0
    optimizations.save(optimization)
    final = fingerprint_saved_evidence(
        [campaigns.load(identifier) for identifier in ids],
        optimization_store=optimizations,
    )
    assert final["manifest_sha256"] != after["manifest_sha256"]


def test_real_cli_exports_manifest_in_report_json(tmp_path, capsys):
    campaigns, optimizations, ids, hardware = build_saved_experiments(tmp_path)
    output = tmp_path / "output"
    command = [
        *ids, "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--hardware-json", str(hardware), "--output-dir", str(output),
        "--iterations", "20",
    ]
    assert main(command) == 0, capsys.readouterr().err
    payload = json.loads((output / "research-report.json").read_text())
    manifest = payload["evidence_manifest"]
    assert len(manifest["sources"]) == 6
    assert len(manifest["manifest_sha256"]) == 64
    assert hashlib.sha256((output / "research-report.json").read_bytes()).hexdigest() in (
        output / "research-report.sha256"
    ).read_text()
