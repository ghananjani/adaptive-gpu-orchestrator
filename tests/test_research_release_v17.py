"""Full disk-backed release manifest regression coverage."""
import hashlib
import json

import pytest
from test_research_disk_e2e_v17 import build_saved_experiments

from adaptive_gpu_orchestrator.research_cli import main as generate
from adaptive_gpu_orchestrator.research_release import (
    create_release_manifest,
    verify_release_manifest,
)
from adaptive_gpu_orchestrator.research_release_cli import main as release_cli

REVISION = "a" * 40


def report_fixture(tmp_path):
    campaigns, optimizations, ids, hardware = build_saved_experiments(tmp_path)
    out = tmp_path / "reports"
    assert generate([
        *ids, "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--hardware-json", str(hardware), "--output-dir", str(out),
        "--iterations", "20",
    ]) == 0
    return out


def test_release_reproducible_and_all_files_checked(tmp_path):
    out = report_fixture(tmp_path)
    path = create_release_manifest(out, release_id="v1.7.0-test", source_revision=REVISION)
    first = path.read_bytes()
    assert create_release_manifest(
        out, release_id="v1.7.0-test", source_revision=REVISION
    ).read_bytes() == first
    trusted_digest = hashlib.sha256(first).hexdigest()
    status = verify_release_manifest(
        path, expected_release_id="v1.7.0-test",
        expected_revision=REVISION, expected_manifest_sha256=trusted_digest,
    )
    assert status["valid"] and status["verified_files"] == 5
    assert release_cli([
        "verify", str(path), "--expected-manifest-sha256", trusted_digest
    ]) == 0
    for artifact in ("research-report.md", "heldout-folds.csv", "research-report.json"):
        original = (out / artifact).read_bytes()
        (out / artifact).write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="release artifact mismatch"):
            verify_release_manifest(path)
        (out / artifact).write_bytes(original)


def test_fails_closed_on_replacement_manifest(tmp_path):
    out = report_fixture(tmp_path)
    path = create_release_manifest(out, release_id="research-1", source_revision=REVISION)
    trusted = hashlib.sha256(path.read_bytes()).hexdigest()
    modified = json.loads(path.read_text())
    modified["source_revision"] = "b" * 40
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="release manifest digest mismatch"):
        verify_release_manifest(path, expected_manifest_sha256=trusted)


def test_rejects_bad_revision_and_missing_reports(tmp_path):
    with pytest.raises(ValueError, match="source_revision"):
        create_release_manifest(tmp_path, release_id="v1", source_revision="main")
    with pytest.raises(FileNotFoundError, match="release artifact missing"):
        create_release_manifest(tmp_path, release_id="v1", source_revision=REVISION)
