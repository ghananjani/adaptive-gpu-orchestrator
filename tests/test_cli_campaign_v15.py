import csv
import json
import sys

from adaptive_gpu_orchestrator.cli import main


def test_campaign_cli_nccl_provenance_and_export(tmp_path, capsys):
    campaigns = tmp_path / "campaigns"
    optimizations = tmp_path / "optimizations"
    studies = tmp_path / "studies"
    nccl_output = tmp_path / "nccl.txt"
    export_path = tmp_path / "campaign.csv"

    nccl_output.write_text(
        "8388608 2097152 float sum -1 10.0 100.0 90.0\n"
        "1073741824 268435456 float sum -1 20.0 140.0 126.0\n",
        encoding="utf-8",
    )

    benchmark = [
        sys.executable,
        "-c",
        (
            "import json; "
            "print(json.dumps({"
            "'throughput_per_s': 123.0, "
            "'p99_latency_ms': 100.0, "
            "'energy_joules': 900.0, "
            "'extras': {'ttft_ms': 55.0}"
            "}))"
        ),
    ]

    code = main(
        [
            "campaign",
            "run",
            "--name",
            "h100-vllm",
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
            "--optimization-store",
            str(optimizations),
            "--study-store",
            str(studies),
            "--store-dir",
            str(campaigns),
            "--nccl-output",
            str(nccl_output),
            "--cuda-toolkit-version",
            "13.0",
            "--nccl-version",
            "2.28.3",
            "--container-image",
            "registry/llm:latest",
            "--container-image-digest",
            "sha256:abc",
            "--",
            *benchmark,
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "NCCL baseline:" in captured.out
    assert "CUDA toolkit: 13.0" in captured.out
    assert "NCCL: 2.28.3" in captured.out

    files = list(campaigns.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    campaign_id = payload["id"]

    assert payload["provenance"]["cuda_toolkit_version"] == "13.0"
    assert payload["provenance"]["nccl_version"] == "2.28.3"
    assert payload["provenance"]["container_image"] == "registry/llm:latest"
    assert payload["nccl_baseline"]["summary"]["sample_count"] == 2
    assert payload["nccl_baseline"]["summary"]["peak_busbw_gbps"] == 126.0

    export_code = main(
        [
            "campaign",
            "export",
            campaign_id,
            "--store-dir",
            str(campaigns),
            "--optimization-store",
            str(optimizations),
            "--output",
            str(export_path),
            "--format",
            "csv",
        ]
    )
    exported = capsys.readouterr()
    assert export_code == 0
    assert "Exported campaign" in exported.out

    with export_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["campaign_group"] == "h100"
    assert rows[0]["nccl_peak_busbw_gbps"] == "126.0"
    assert rows[0]["cuda_toolkit_version"] == "13.0"
    assert rows[0]["container_image_digest"] == "sha256:abc"
