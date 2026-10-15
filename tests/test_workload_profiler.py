from adaptive_gpu_orchestrator.models import GPUDevice
from adaptive_gpu_orchestrator.profiling import WorkloadProfiler


class FakeProvider:
    def sample(self):
        return [
            GPUDevice(
                index=0,
                name="NVIDIA H200",
                memory_total_mib=140000,
                memory_used_mib=70000,
                utilization_gpu_pct=80,
                temperature_c=62,
                power_draw_w=500,
                power_limit_w=700,
            )
        ]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_profiles_timed_observation_window():
    clock = FakeClock()
    profiler = WorkloadProfiler(
        provider=FakeProvider(),
        clock=clock,
        sleeper=clock.sleep,
    )

    profile = profiler.run(
        name="h200-observation",
        duration_s=2.0,
        interval_s=1.0,
        tags={"cluster": "test"},
    )

    assert profile.duration_s == 2.0
    assert len(profile.samples) == 3
    assert profile.summary()["gpu_util_mean_pct"] == 80.0
    assert profile.tags["cluster"] == "test"
