from adaptive_gpu_orchestrator.correlation import CorrelationStore, correlate_optimization

from .test_correlation import make_optimization, make_profile


def test_correlation_store_round_trip(tmp_path):
    report = correlate_optimization(make_optimization(), make_profile())
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
        store.save(correlate_optimization(make_optimization(), make_profile()))

    assert len(store.list()) == 2
