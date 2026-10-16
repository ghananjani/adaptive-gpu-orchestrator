from adaptive_gpu_orchestrator.autopilot import AutopilotPolicy, decide_autopilot
from adaptive_gpu_orchestrator.controller import SchedulerBackend
from adaptive_gpu_orchestrator.evidence import (
    EvidenceBundleStore,
    RuntimeProvenance,
    build_evidence_bundle,
    verify_bundle,
)
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)
from adaptive_gpu_orchestrator.validation import ValidationPolicy


def optimization(identifier, tp, throughput):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=100.0,
            energy_joules=1000.0,
        ),
        score=1.0,
    )
    return OptimizationResult(
        id=identifier,
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
    )


def test_evidence_store_round_trip_preserves_integrity(tmp_path):
    baseline = optimization("baseline", 1, 100.0)
    candidate = optimization("candidate", 2, 120.0)
    report = decide_autopilot(
        baseline=baseline,
        candidate=candidate,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=2,
        policy=AutopilotPolicy(
            validation=ValidationPolicy(min_throughput_gain_pct=5)
        ),
    )
    bundle = build_evidence_bundle(
        autopilot=report,
        baseline=baseline,
        candidate=candidate,
        provenance=RuntimeProvenance(
            python_version="3.12.0",
            implementation="cpython",
            platform="Linux-test",
            machine="x86_64",
        ),
    )

    store = EvidenceBundleStore(tmp_path)
    path = store.save(bundle)
    loaded = store.load(bundle.id)

    assert path.exists()
    assert loaded.id == bundle.id
    assert loaded.sha256 == bundle.sha256
    assert verify_bundle(loaded).valid is True
    assert len(store.list()) == 1
