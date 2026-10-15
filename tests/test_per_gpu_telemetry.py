from adaptive_gpu_orchestrator.models import GPUDevice
from adaptive_gpu_orchestrator.telemetry import (
    DeviceTelemetry,
    TelemetrySample,
    TelemetrySampler,
)


class FakeProvider:
    def sample(self):
        return [
            GPUDevice(
                index=0,
                name="H100",
                uuid="GPU-0",
                memory_total_mib=80_000,
                memory_used_mib=40_000,
                utilization_gpu_pct=90.0,
                utilization_memory_pct=55.0,
                temperature_c=70.0,
                power_draw_w=500.0,
                power_limit_w=700.0,
            ),
            GPUDevice(
                index=1,
                name="H100",
                uuid="GPU-1",
                memory_total_mib=80_000,
                memory_used_mib=20_000,
                utilization_gpu_pct=50.0,
                utilization_memory_pct=30.0,
                temperature_c=68.0,
                power_draw_w=400.0,
                power_limit_w=700.0,
            ),
        ]


def test_sampler_retains_per_gpu_telemetry():
    sample = TelemetrySampler(FakeProvider()).sample_once(1.5)

    assert sample.gpu_count == 2
    assert sample.gpu_util_mean_pct == 70.0
    assert len(sample.devices) == 2
    assert sample.devices[0].uuid == "GPU-0"
    assert sample.devices[0].memory_pressure_pct == 50.0
    assert sample.devices[1].gpu_util_pct == 50.0


def test_telemetry_sample_old_payload_remains_compatible():
    sample = TelemetrySample.from_dict(
        {
            "offset_s": 0.0,
            "gpu_count": 1,
            "gpu_util_mean_pct": 80.0,
            "gpu_util_std_pct": 0.0,
            "memory_pressure_mean_pct": 50.0,
            "temperature_max_c": 70.0,
            "power_ratio_max_pct": 80.0,
            "power_draw_total_w": 450.0,
        }
    )

    assert sample.devices == ()


def test_device_telemetry_round_trip():
    device = DeviceTelemetry(
        index=3,
        uuid="GPU-3",
        gpu_util_pct=88.0,
        memory_copy_util_pct=44.0,
        memory_pressure_pct=75.0,
        memory_used_mib=60_000,
        memory_total_mib=80_000,
        temperature_c=72.0,
        power_draw_w=520.0,
        power_limit_w=700.0,
    )

    assert DeviceTelemetry.from_dict(device.to_dict()) == device
