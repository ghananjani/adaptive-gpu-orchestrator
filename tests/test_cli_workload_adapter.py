import sys

from adaptive_gpu_orchestrator.cli import main
from adaptive_gpu_orchestrator.optimizer import OptimizationStore


def test_optimize_with_vllm_adapter_persists_native_metrics(tmp_path, capsys):
    script = (
        "import json, os; "
        "tp=float(os.environ['AGO_TP']); "
        "print(json.dumps({"
        "'request_throughput': tp * 5, "
        "'output_throughput': tp * 500, "
        "'p99_ttft_ms': 100 / tp, "
        "'p99_itl_ms': 12 / tp"
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
            "--adapter",
            "vllm",
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

    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    result = OptimizationStore(tmp_path).load(paths[0])
    assert result.best is not None
    assert result.best.metrics.throughput_per_s == 1000.0
    assert result.best.metrics.p99_latency_ms == 50.0
    assert result.best.metrics.extras["p99_itl_ms"] == 6.0


def test_optimize_with_pytorch_adapter_uses_training_metrics(tmp_path, capsys):
    script = (
        "import json, os; "
        "tp=float(os.environ['AGO_TP']); "
        "print(json.dumps({"
        "'samples_per_s': tp * 32, "
        "'p99_step_time_ms': 120 / tp, "
        "'mean_step_time_ms': 100 / tp, "
        "'communication_time_ms': 20 / tp"
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
            "--adapter",
            "pytorch-distributed",
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
