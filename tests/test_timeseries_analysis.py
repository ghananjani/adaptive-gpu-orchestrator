from adaptive_gpu_orchestrator.profiling import WorkloadProfile
from adaptive_gpu_orchestrator.telemetry import (
    DeviceTelemetry,
    TelemetrySample,
    parse_dcgm_exporter,
)
from adaptive_gpu_orchestrator.timeseries import PhaseWindow, analyze_timeseries


def sample(offset, util0, util1, memory0=70.0, memory1=70.0):
    devices = (
        DeviceTelemetry(
            index=0,
            gpu_util_pct=util0,
            memory_pressure_pct=memory0,
            power_draw_w=500.0,
            temperature_c=72.0,
        ),
        DeviceTelemetry(
            index=1,
            gpu_util_pct=util1,
            memory_pressure_pct=memory1,
            power_draw_w=400.0,
            temperature_c=70.0,
        ),
    )
    return TelemetrySample(
        offset_s=offset,
        gpu_count=2,
        gpu_util_mean_pct=(util0 + util1) / 2,
        gpu_util_std_pct=None,
        memory_pressure_mean_pct=(memory0 + memory1) / 2,
        temperature_max_c=72.0,
        power_ratio_max_pct=None,
        power_draw_total_w=900.0,
        devices=devices,
    )


def test_timeseries_detects_straggler_and_summarizes_phases():
    profile = WorkloadProfile(
        id="profile-v08",
        name="training",
        duration_s=4.0,
        samples=[
            sample(0.0, 95.0, 55.0),
            sample(1.0, 94.0, 54.0),
            sample(2.0, 93.0, 53.0, memory1=96.0),
            sample(3.0, 92.0, 52.0, memory1=97.0),
        ],
    )
    dcgm = parse_dcgm_exporter(
        """
DCGM_FI_DEV_GPU_UTIL{gpu="0"} 94
DCGM_FI_DEV_GPU_UTIL{gpu="1"} 54
"""
    )

    report = analyze_timeseries(
        profile,
        phases=[
            PhaseWindow("warmup", 0.0, 2.0),
            PhaseWindow("steady", 2.0, 4.0),
        ],
        dcgm_frames=dcgm,
        straggler_gap_pct_points=15.0,
    )

    assert report.straggler_gpu_indices == [1]
    assert report.gpu_summaries[1].memory_pressure_max_pct == 97.0
    assert report.phases[0].gpu_util_by_index[0] == 94.5
    assert report.phases[1].gpu_util_by_index[1] == 52.5
    assert any("Persistent utilization straggler" in item for item in report.findings)
    assert any("DCGM GPU-utilization imbalance" in item for item in report.findings)


def test_timeseries_old_profile_reports_missing_per_gpu_data():
    profile = WorkloadProfile(
        id="old",
        name="old-profile",
        samples=[
            TelemetrySample(
                offset_s=0.0,
                gpu_count=2,
                gpu_util_mean_pct=70.0,
                gpu_util_std_pct=5.0,
                memory_pressure_mean_pct=60.0,
                temperature_max_c=70.0,
                power_ratio_max_pct=80.0,
                power_draw_total_w=700.0,
            )
        ],
    )

    report = analyze_timeseries(profile)

    assert report.gpu_summaries == []
    assert any("aggregate-only telemetry" in item for item in report.findings)
