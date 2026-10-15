from adaptive_gpu_orchestrator.models import Bottleneck
from adaptive_gpu_orchestrator.profiling import WorkloadProfile, diagnose_profile
from adaptive_gpu_orchestrator.telemetry.sampler import TelemetrySample


def make_sample(
    *,
    gpus=2,
    util=70.0,
    imbalance=5.0,
    memory=50.0,
    temp=65.0,
):
    return TelemetrySample(
        offset_s=0.0,
        gpu_count=gpus,
        gpu_util_mean_pct=util,
        gpu_util_std_pct=imbalance,
        memory_pressure_mean_pct=memory,
        temperature_max_c=temp,
        power_ratio_max_pct=70.0,
        power_draw_total_w=800.0,
    )


def test_profile_detects_memory_pressure():
    profile = WorkloadProfile(
        name="memory-heavy",
        samples=[
            make_sample(util=65, memory=95),
            make_sample(util=68, memory=94),
        ],
    )

    report = diagnose_profile(profile)

    assert report.bottleneck is Bottleneck.MEMORY_PRESSURE
    assert report.severity == "high"


def test_profile_detects_load_imbalance():
    profile = WorkloadProfile(
        name="imbalanced",
        samples=[
            make_sample(imbalance=25),
            make_sample(imbalance=30),
        ],
    )

    assert diagnose_profile(profile).bottleneck is Bottleneck.LOAD_IMBALANCE


def test_empty_profile_reports_no_gpu():
    profile = WorkloadProfile(name="empty")

    assert diagnose_profile(profile).bottleneck is Bottleneck.NO_GPU
