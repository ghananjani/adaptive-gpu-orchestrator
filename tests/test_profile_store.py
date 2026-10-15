from adaptive_gpu_orchestrator.profiling import ExperimentStore, WorkloadProfile
from adaptive_gpu_orchestrator.telemetry.sampler import TelemetrySample


def sample():
    return TelemetrySample(
        offset_s=0.0,
        gpu_count=2,
        gpu_util_mean_pct=75.0,
        gpu_util_std_pct=5.0,
        memory_pressure_mean_pct=60.0,
        temperature_max_c=64.0,
        power_ratio_max_pct=70.0,
        power_draw_total_w=900.0,
    )


def test_profile_round_trip(tmp_path):
    profile = WorkloadProfile(
        id="abc123",
        name="llama70b",
        command=["python", "serve.py"],
        samples=[sample()],
        duration_s=12.5,
        exit_code=0,
        tags={"model": "llama70b"},
    )
    store = ExperimentStore(tmp_path)

    path = store.save(profile)
    loaded = store.load("abc123")

    assert path.exists()
    assert loaded.id == profile.id
    assert loaded.command == profile.command
    assert loaded.tags == profile.tags
    assert loaded.summary()["gpu_util_mean_pct"] == 75.0


def test_store_lists_saved_profiles(tmp_path):
    store = ExperimentStore(tmp_path)
    store.save(WorkloadProfile(id="one", name="first", samples=[sample()]))
    store.save(WorkloadProfile(id="two", name="second", samples=[sample()]))

    assert {profile.id for profile in store.list()} == {"one", "two"}
