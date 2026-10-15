import pytest

from adaptive_gpu_orchestrator.correlation import correlate_optimization
from adaptive_gpu_orchestrator.models import NCCLResult
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)
from adaptive_gpu_orchestrator.profiling import WorkloadProfile
from adaptive_gpu_orchestrator.telemetry import TelemetrySample


def make_optimization():
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=2, batch_size=4),
        metrics=TrialMetrics(
            throughput_per_s=100.0,
            p99_latency_ms=80.0,
            extras={"communication_fraction_pct": 30.0},
        ),
        score=1.0,
    )
    return OptimizationResult(
        id="opt123",
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
    )


def make_profile():
    return WorkloadProfile(
        id="profile123",
        name="training",
        duration_s=10.0,
        samples=[
            TelemetrySample(
                offset_s=0.0,
                gpu_count=2,
                gpu_util_mean_pct=50.0,
                gpu_util_std_pct=18.0,
                memory_pressure_mean_pct=91.0,
                temperature_max_c=84.0,
                power_ratio_max_pct=90.0,
                power_draw_total_w=400.0,
            ),
            TelemetrySample(
                offset_s=1.0,
                gpu_count=2,
                gpu_util_mean_pct=55.0,
                gpu_util_std_pct=20.0,
                memory_pressure_mean_pct=93.0,
                temperature_max_c=86.0,
                power_ratio_max_pct=92.0,
                power_draw_total_w=400.0,
            ),
        ],
    )


def test_correlation_combines_telemetry_adapter_metrics_and_nccl():
    nccl = [
        NCCLResult(
            operation="all_reduce",
            size_bytes=8_388_608,
            time_us=100.0,
            algbw_gbps=80.0,
            busbw_gbps=70.0,
        ),
        NCCLResult(
            operation="all_reduce",
            size_bytes=16_777_216,
            time_us=110.0,
            algbw_gbps=90.0,
            busbw_gbps=80.0,
        ),
    ]

    report = correlate_optimization(
        make_optimization(),
        make_profile(),
        nccl_results=nccl,
    )

    assert report.telemetry["gpu_util_mean_pct"] == 52.5
    assert report.telemetry["estimated_energy_joules"] == 4000.0
    assert report.telemetry["energy_per_throughput_unit_j"] == 4.0
    assert report.nccl["busbw_mean_gbps"] == 75.0
    assert report.adapter_metrics["communication_fraction_pct"] == 30.0
    assert any("underfeeding accelerators" in item for item in report.findings)
    assert any("memory headroom is limited" in item for item in report.findings)
    assert any("NCCL mean bus bandwidth baseline" in item for item in report.findings)


def test_correlation_requires_feasible_best_trial():
    optimization = OptimizationResult(
        id="empty",
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[],
        best=None,
    )

    with pytest.raises(ValueError):
        correlate_optimization(optimization, make_profile())
