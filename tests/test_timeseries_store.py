from adaptive_gpu_orchestrator.timeseries import (
    GPUTimeSeriesSummary,
    PhaseSummary,
    TimeSeriesReport,
    TimeSeriesStore,
)


def test_timeseries_store_round_trip(tmp_path):
    report = TimeSeriesReport(
        id="ts123",
        profile_id="profile123",
        gpu_summaries=[
            GPUTimeSeriesSummary(
                gpu_index=0,
                sample_count=3,
                gpu_util_mean_pct=90.0,
                straggler_gap_pct_points=0.0,
            )
        ],
        phases=[
            PhaseSummary(
                name="steady",
                start_s=1.0,
                end_s=3.0,
                sample_count=2,
                gpu_util_mean_pct=90.0,
                gpu_util_by_index={0: 90.0},
            )
        ],
        dcgm_summary={0: {"DCGM_FI_DEV_GPU_UTIL": 91.0}},
        findings=["healthy"],
    )
    store = TimeSeriesStore(tmp_path)

    path = store.save(report)
    loaded = store.load("ts123")

    assert path.exists()
    assert loaded.id == report.id
    assert loaded.profile_id == "profile123"
    assert loaded.gpu_summaries == report.gpu_summaries
    assert loaded.phases == report.phases
    assert loaded.dcgm_summary == report.dcgm_summary
