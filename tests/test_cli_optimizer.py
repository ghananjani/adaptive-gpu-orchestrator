import sys

from adaptive_gpu_orchestrator.cli import main


def test_optimize_dry_run(capsys):
    code = main(
        [
            "optimize",
            "--gpus",
            "4",
            "--tp",
            "1,2,4",
            "--pp",
            "1",
            "--dp",
            "1",
            "--batch",
            "1",
            "--concurrency",
            "1",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "tp4-pp1-dp1-b1-c1" in captured.out


def test_optimize_runs_benchmark_and_persists_result(tmp_path, capsys):
    script = (
        "import json, os; "
        "tp=float(os.environ['AGO_TP']); "
        "print(json.dumps({"
        "'throughput_per_s': tp * 100, "
        "'p99_latency_ms': 100 / tp, "
        "'energy_joules': 1000"
        "}))"
    )
    code = main(
        [
            "optimize",
            "--gpus",
            "2",
            "--tp",
            "1,2",
            "--pp",
            "1",
            "--dp",
            "1",
            "--batch",
            "1",
            "--concurrency",
            "1",
            "--store-dir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            script,
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "tp2-pp1-dp1-b1-c1" in captured.out
    assert list(tmp_path.glob("*.json"))
