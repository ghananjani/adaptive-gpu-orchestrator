import json
import sys

from adaptive_gpu_orchestrator.cli import main


def benchmark_command(flag_path):
    script = f"""
import json
import os
from pathlib import Path

flag = Path({str(flag_path)!r})
replicate = os.environ["AGO_REPLICATE_INDEX"]
if replicate == "2" and not flag.exists():
    flag.write_text("seen", encoding="utf-8")
    raise SystemExit(1)

print(json.dumps({{
    "throughput_per_s": 100.0,
    "p99_latency_ms": 100.0,
    "energy_joules": 1000.0
}}))
"""
    return [sys.executable, "-c", script]


def test_campaign_manifest_run_and_resume_workflow(tmp_path, capsys):
    campaigns = tmp_path / "campaigns"
    optimizations = tmp_path / "optimizations"
    studies = tmp_path / "studies"
    manifest_path = tmp_path / "campaign-manifest.json"
    flag = tmp_path / "replicate-two.flag"

    run_code = main(
        [
            "campaign",
            "run",
            "--name",
            "resume-workflow",
            "--group",
            "h100",
            "--repeats",
            "2",
            "--gpus",
            "1",
            "--tp",
            "1",
            "--pp",
            "1",
            "--dp",
            "1",
            "--batch",
            "1",
            "--concurrency",
            "1",
            "--randomize-candidates",
            "--random-seed",
            "17",
            "--optimization-store",
            str(optimizations),
            "--study-store",
            str(studies),
            "--store-dir",
            str(campaigns),
            "--",
            *benchmark_command(flag),
        ]
    )
    first_output = capsys.readouterr()

    assert run_code == 0
    assert "Replicates: 1/2" in first_output.out
    assert "pending=[2]" in first_output.out

    campaign_files = list(campaigns.glob("*.json"))
    assert len(campaign_files) == 1
    original_payload = json.loads(
        campaign_files[0].read_text(encoding="utf-8")
    )
    original_id = original_payload["id"]
    replicate_one_id = next(
        run["optimization_id"]
        for run in original_payload["runs"]
        if run["replicate"] == 1
    )

    manifest_code = main(
        [
            "campaign",
            "manifest",
            original_id,
            "--output",
            str(manifest_path),
            "--store-dir",
            str(campaigns),
        ]
    )
    manifest_output = capsys.readouterr()

    assert manifest_code == 0
    assert manifest_path.exists()
    assert "sha256=" in manifest_output.out

    manifest_payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    assert manifest_payload["spec"]["randomize_candidates"] is True
    assert manifest_payload["spec"]["random_seed"] == 17
    assert len(manifest_payload["sha256"]) == 64

    manifest_run_code = main(
        [
            "campaign",
            "run",
            "--manifest",
            str(manifest_path),
            "--optimization-store",
            str(optimizations),
            "--study-store",
            str(studies),
            "--store-dir",
            str(campaigns),
        ]
    )
    manifest_run_output = capsys.readouterr()

    assert manifest_run_code == 0
    assert "Replicates: 2/2" in manifest_run_output.out
    assert "pending=none" in manifest_run_output.out

    resume_code = main(
        [
            "campaign",
            "resume",
            original_id,
            "--optimization-store",
            str(optimizations),
            "--study-store",
            str(studies),
            "--store-dir",
            str(campaigns),
        ]
    )
    resumed_output = capsys.readouterr()

    assert resume_code == 0
    assert "Replicates: 2/2" in resumed_output.out
    assert "Resume count: 1" in resumed_output.out
    assert "pending=none" in resumed_output.out

    resumed_payload = json.loads(
        (campaigns / f"{original_id}.json").read_text(encoding="utf-8")
    )
    assert resumed_payload["resume_count"] == 1
    assert resumed_payload["pending_replicates"] == []
    assert len(resumed_payload["resume_provenance"]) == 1

    replicate_one_after = next(
        run["optimization_id"]
        for run in resumed_payload["runs"]
        if run["replicate"] == 1
    )
    assert replicate_one_after == replicate_one_id

    replicate_two = next(
        run
        for run in resumed_payload["runs"]
        if run["replicate"] == 2
    )
    assert replicate_two["status"] == "completed"
    assert replicate_two["candidate_order"] == ["tp1-pp1-dp1-b1-c1"]


def test_campaign_run_requires_no_cli_overrides_with_manifest(tmp_path, capsys):
    manifest_path = tmp_path / "missing.json"

    code = main(
        [
            "campaign",
            "run",
            "--manifest",
            str(manifest_path),
            "--tag",
            "hardware=h100",
        ]
    )
    output = capsys.readouterr()

    assert code == 2
    assert "cannot override" in output.err or "not found" in output.err
