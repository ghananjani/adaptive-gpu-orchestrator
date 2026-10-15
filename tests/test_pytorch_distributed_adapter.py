from adaptive_gpu_orchestrator.adapters import PyTorchDistributedMetricsAdapter


def test_pytorch_adapter_normalizes_training_metrics():
    payload = {
        "samples_per_s": 64.0,
        "p99_step_time_ms": 120.0,
        "mean_step_time_ms": 100.0,
        "communication_time_ms": 25.0,
        "compute_time_ms": 70.0,
        "gpu_util_pct": 95.0,
        "memory_pressure_pct": 88.0,
        "phases": {
            "forward_ms": 22.0,
            "backward_ms": 41.0,
        },
    }

    metrics = PyTorchDistributedMetricsAdapter().parse(payload)

    assert metrics.throughput_per_s == 64.0
    assert metrics.p99_latency_ms == 120.0
    assert metrics.gpu_util_pct == 95.0
    assert metrics.extras["communication_fraction_pct"] == 25.0
    assert metrics.extras["compute_fraction_pct"] == 70.0
    assert metrics.extras["phases.forward_ms"] == 22.0


def test_pytorch_adapter_prefers_token_throughput_when_available():
    metrics = PyTorchDistributedMetricsAdapter().parse(
        {
            "tokens_per_s": 10000.0,
            "samples_per_s": 64.0,
            "step_time_p99_ms": 90.0,
        }
    )

    assert metrics.throughput_per_s == 10000.0
    assert metrics.p99_latency_ms == 90.0
