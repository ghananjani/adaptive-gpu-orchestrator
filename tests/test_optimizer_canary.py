import sys

from adaptive_gpu_orchestrator.optimizer.canary import CommandCanaryRunner
from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig


def test_command_canary_injects_configuration_and_parses_json():
    script = (
        "import json, os; "
        "print(json.dumps({"
        "'throughput_per_s': float(os.environ['AGO_TP']) * 100, "
        "'p99_latency_ms': 50.0, "
        "'energy_joules': 1000.0"
        "}))"
    )
    runner = CommandCanaryRunner([sys.executable, "-c", script], timeout_s=10)
    metrics = runner.run(
        ParallelismConfig(
            tensor_parallel=4,
            pipeline_parallel=1,
            data_parallel=1,
            batch_size=1,
            concurrency=1,
        )
    )

    assert metrics.exit_code == 0
    assert metrics.throughput_per_s == 400
    assert metrics.p99_latency_ms == 50


def test_command_canary_nonzero_exit_becomes_failed_trial():
    runner = CommandCanaryRunner([sys.executable, "-c", "raise SystemExit(7)"])
    metrics = runner.run(ParallelismConfig())

    assert metrics.exit_code == 7
