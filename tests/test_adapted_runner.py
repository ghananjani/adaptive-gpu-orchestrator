import sys

from adaptive_gpu_orchestrator.adapters import (
    AdaptedCommandCanaryRunner,
    VLLMMetricsAdapter,
)
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig


def test_adapted_runner_sets_adapter_environment_and_parses_native_json():
    script = (
        "import json, os; "
        "assert os.environ['AGO_METRICS_ADAPTER'] == 'vllm'; "
        "print(json.dumps({"
        "'output_throughput': 500.0, "
        "'p99_ttft_ms': 80.0"
        "}))"
    )
    runner = AdaptedCommandCanaryRunner(
        adapter=VLLMMetricsAdapter(),
        command=[sys.executable, "-c", script],
        timeout_s=10,
    )

    metrics = runner.run(
        ParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=1,
            data_parallel=1,
            batch_size=1,
            concurrency=8,
        )
    )

    assert metrics.exit_code == 0
    assert metrics.throughput_per_s == 500.0
    assert metrics.p99_latency_ms == 80.0
    assert metrics.duration_s is not None


def test_adapted_runner_nonzero_exit_is_failed_trial():
    runner = AdaptedCommandCanaryRunner(
        adapter=VLLMMetricsAdapter(),
        command=[sys.executable, "-c", "raise SystemExit(9)"],
    )

    metrics = runner.run(ParallelismConfig())

    assert metrics.exit_code == 9
