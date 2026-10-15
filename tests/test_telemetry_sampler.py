from adaptive_gpu_orchestrator.models import GPUDevice
from adaptive_gpu_orchestrator.telemetry.providers import NvidiaSMIProvider
from adaptive_gpu_orchestrator.telemetry.sampler import TelemetrySampler


class FakeProvider:
    def sample(self):
        return [
            GPUDevice(
                index=0,
                name="NVIDIA H100",
                memory_total_mib=80000,
                memory_used_mib=40000,
                utilization_gpu_pct=50,
                temperature_c=60,
                power_draw_w=400,
                power_limit_w=700,
            ),
            GPUDevice(
                index=1,
                name="NVIDIA H100",
                memory_total_mib=80000,
                memory_used_mib=48000,
                utilization_gpu_pct=70,
                temperature_c=64,
                power_draw_w=420,
                power_limit_w=700,
            ),
        ]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_collects_deterministic_time_series():
    clock = FakeClock()
    sampler = TelemetrySampler(
        FakeProvider(),
        clock=clock,
        sleeper=clock.sleep,
    )

    samples = sampler.collect(duration_s=2.0, interval_s=1.0)

    assert [sample.offset_s for sample in samples] == [0.0, 1.0, 2.0]
    assert all(sample.gpu_count == 2 for sample in samples)
    assert samples[0].gpu_util_mean_pct == 60
    assert samples[0].memory_pressure_mean_pct == 55
    assert samples[0].power_draw_total_w == 820


def test_nvidia_smi_provider_degrades_to_empty_sample():
    class BrokenSMI:
        def query_gpus(self):
            from adaptive_gpu_orchestrator.discovery import NvidiaSMIError

            raise NvidiaSMIError("no device")

    provider = NvidiaSMIProvider(BrokenSMI())
    assert provider.sample() == []
