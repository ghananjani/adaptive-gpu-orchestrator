import json
import sys

from adaptive_gpu_orchestrator.cli import main


def write_fake_nccl_binary(path):
    path.write_text(
        (
            f"#!{sys.executable}\n"
            "print('8388608 2097152 float sum -1 100.0 80.0 70.0')\n"
            "print('16777216 4194304 float sum -1 110.0 90.0 80.0')\n"
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def benchmark_command():
    return [
        sys.executable,
        "-c",
        (
            "import json; "
            "print(json.dumps({'throughput_per_s': 100.0, "
            "'p99_latency_ms': 100.0, 'energy_joules': 1000.0}))"
        ),
    ]


def test_campaign_cli_captures_nccl_baseline_and_manifest(tmp_path, capsys):
    campaigns = tmp_path / "campaigns"
    optimizations = tmp_path / "optimizations"
    studies = tmp_path / "studies"
    manifest = tmp_path / "campaign.json"
    nccl_binary = tmp_path / "all_reduce_perf"
    write_fake_nccl_binary(nccl_binary)

    code = main(
        [
            "campaign",
            "run",
            "--name",
            "h100-nccl",
            "--group",
            "h100",
            "--repeats",
            "1",
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
            "--nccl-baseline",
            "--nccl-binary",
            str(nccl_binary),
            "--nccl-gpus",
            "1",
            "--optimization-store",
            str(optimizations),
            "--study-store",
            str(studies),
            "--store-dir",
            str(campaigns),
            "--",
            *benchmark_command(),
        ]
    )
    output = capsys.readouterr()

    assert code == 0
    assert "NCCL communication baseline:" in output.out
    assert "75.00 GB/s" in output.out

    paths = list(campaigns.glob("*.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    campaign_id = payload["id"]

    baseline = payload["communication_baseline"]
    assert baseline["summary"]["busbw_mean_gbps"] == 75.0
    assert baseline["summary"]["largest_message_busbw_gbps"] == 80.0
    assert len(baseline["sha256"]) == 64
    assert payload["spec"]["nccl_baseline"]["gpus"] == 1
    assert payload["spec"]["nccl_baseline"]["binary"] == str(nccl_binary)

    manifest_code = main(
        [
            "campaign",
            "manifest",
            campaign_id,
            "--output",
            str(manifest),
            "--store-dir",
            str(campaigns),
        ]
    )
    capsys.readouterr()

    assert manifest_code == 0
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["spec"]["nccl_baseline"]["gpus"] == 1
    assert manifest_payload["spec"]["nccl_baseline"]["binary"] == str(nccl_binary)
