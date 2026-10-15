from adaptive_gpu_orchestrator.studies import (
    MetricSummary,
    StudyGroupSummary,
    StudyReport,
    StudyStore,
)


def test_study_store_round_trip(tmp_path):
    report = StudyReport(
        id="study123",
        groups=[
            StudyGroupSummary(
                name="h100",
                optimization_ids=("a1", "a2"),
                metrics={
                    "throughput_per_s": MetricSummary(
                        metric="throughput_per_s",
                        n=2,
                        mean=100.0,
                        minimum=95.0,
                        maximum=105.0,
                        stdev=7.071,
                        ci95_low=36.47,
                        ci95_high=163.53,
                    )
                },
                config_counts={"tp2-pp1-dp1-b1-c1": 2},
                dominant_config="tp2-pp1-dp1-b1-c1",
                config_stability_fraction=1.0,
            )
        ],
        findings=["example"],
    )

    store = StudyStore(tmp_path)
    path = store.save(report)
    loaded = store.load("study123")

    assert path.exists()
    assert loaded.id == report.id
    assert loaded.groups == report.groups
    assert loaded.findings == ["example"]
