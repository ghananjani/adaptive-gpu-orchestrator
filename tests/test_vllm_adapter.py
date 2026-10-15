from adaptive_gpu_orchestrator.adapters import VLLMMetricsAdapter


def test_vllm_adapter_normalizes_serving_metrics():
    payload = {
        "completed": 100,
        "duration": 10.0,
        "request_throughput": 10.0,
        "output_throughput": 900.0,
        "total_token_throughput": 1200.0,
        "p99_ttft_ms": 150.0,
        "p99_tpot_ms": 12.0,
        "p99_itl_ms": 13.0,
        "gpu_util_pct": 92.0,
        "memory_pressure_pct": 86.0,
    }

    metrics = VLLMMetricsAdapter().parse(payload)

    assert metrics.throughput_per_s == 1200.0
    assert metrics.p99_latency_ms == 150.0
    assert metrics.duration_s == 10.0
    assert metrics.gpu_util_pct == 92.0
    assert metrics.extras["request_throughput"] == 10.0
    assert metrics.extras["p99_tpot_ms"] == 12.0
    assert metrics.extras["p99_itl_ms"] == 13.0


def test_vllm_adapter_prefers_end_to_end_p99_when_available():
    metrics = VLLMMetricsAdapter().parse(
        {
            "request_throughput": 5.0,
            "p99_ttft_ms": 100.0,
            "p99_e2el_ms": 450.0,
        }
    )

    assert metrics.throughput_per_s == 5.0
    assert metrics.p99_latency_ms == 450.0
