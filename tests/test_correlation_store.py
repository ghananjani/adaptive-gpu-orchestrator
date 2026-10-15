from adaptive_gpu_orchestrator.correlation import CorrelationStore, correlate_optimization
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


def make_report():
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=2),
        metrics=TrialMetrics(throughput_per_s=100.0),
        score=1.0,
    )
    optimization = OptimizationResult(
        id="opt123",
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
    )
    profile = WorkloadProfile(
        id="profile123",
        name="training",
        duration_s=2.0,
        samples=[
            TelemetrySample(
                offset_s=0.0,
                gpu_count=2,
                gpu_util_mean_pct=70.0,
                gpu_util_std_pct=5.0,
                memory_pressure_mean_pct=60.0,
                temperature_max_c=70.0,
                power_ratio_max_pct=80.0,
                power_draw_total_w=300.0,
            )
        ],
    )
    return correlate_optimization(optimization, profile)


def test_correlation_store_round_trip(tmp_path):
    report = make_report()
    store = CorrelationStore(tmp_path)

    path = store.save(report)
    loaded = store.load(report.id)

    assert path.exists()
    assert loaded.id == report.id
    assert loaded.optimization_id == "opt123"
    assert loaded.profile_id == "profile123"
    assert loaded.config == report.config
    assert loaded.telemetry == report.telemetry


def test_correlation_store_lists_reports(tmp_path):
    store = CorrelationStore(tmp_path)
    for _ in range(2):
        store.save(make_report())

    assert len(store.list()) == 2
