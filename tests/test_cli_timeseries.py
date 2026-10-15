from adaptive_gpu_orchestrator.cli import main
from adaptive_gpu_orchestrator.profiling import ExperimentStore, WorkloadProfile
from adaptive_gpu_orchestrator.telemetry import DeviceTelemetry, TelemetrySample


def telemetry_sample(offset, util0, util1):
    return TelemetrySample(
        offset_s=offset,
        gpu_count=2,
        gpu_util_mean_pct=(util0 + util1) / 2,
        gpu_util_std_pct=None,
        memory_pressure_mean_pct=70.0,
        temperature_max_c=72.0,
        power_ratio_max_pct=None,
        power_draw_total_w=900.0,
        devices=(
            DeviceTelemetry(
                index=0,
                gpu_util_pct=util0,
                memory_pressure_pct=70.0,
                power_draw_w=500.0,
                temperature_c=72.0,
            ),
            DeviceTelemetry(
                index=1,
                gpu_util_pct=util1,
                memory_pressure_pct=70.0,
                power_draw_w=400.0,
                temperature_c=70.0,
            ),
        ),
    )


def test_timeseries_cli_analyzes_profile_dcgm_and_phases(tmp_path, capsys):
    experiments = tmp_path / "experiments"
    reports = tmp_path / "timeseries"

    ExperimentStore(experiments).save(
        WorkloadProfile(
            id="profile-ts",
            name="training",
            duration_s=3.0,
            samples=[
                telemetry_sample(0.0, 95.0, 55.0),
                telemetry_sample(1.0, 94.0, 54.0),
                telemetry_sample(2.0, 93.0, 53.0),
            ],
        )
    )

    dcgm = tmp_path / "dcgm.prom"
    dcgm.write_text(
        'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 94\n'
        'DCGM_FI_DEV_GPU_UTIL{gpu="1"} 54\n',
        encoding="utf-8",
    )

    code = main(
        [
            "timeseries",
            "analyze",
            "--profile",
            "profile-ts",
            "--experiment-store",
            str(experiments),
            "--phase",
            "warmup:0:1",
            "--phase",
            "steady:1:3",
            "--dcgm-file",
            str(dcgm),
            "--store-dir",
            str(reports),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "stragglers=[1]" in captured.out
    assert "DCGM GPU-utilization imbalance" in captured.out
    assert list(reports.glob("*.json"))

    list_code = main(
        [
            "timeseries",
            "list",
            "--store-dir",
            str(reports),
        ]
    )
    listed = capsys.readouterr()
    assert list_code == 0
    assert "profile=profile-ts" in listed.out
