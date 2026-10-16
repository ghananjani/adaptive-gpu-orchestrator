import pytest

from adaptive_gpu_orchestrator.autopilot import AutopilotPolicy, decide_autopilot
from adaptive_gpu_orchestrator.controller import SchedulerBackend
from adaptive_gpu_orchestrator.evidence import (
    RuntimeProvenance,
    build_evidence_bundle,
    sha256_payload,
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
            memory_pressure_pct=50.0,
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


def autopilot_report():
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
    return baseline, candidate, report


def provenance():
    return RuntimeProvenance(
        python_version="3.12.0",
        implementation="cpython",
        platform="Linux-test",
        machine="x86_64",
        package_version="1.1.0a0",
        git_commit="abc123",
        environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        tags={"gpu": "h100"},
    )


def test_canonical_hash_is_independent_of_dict_key_order():
    left = {"b": 2, "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": 2}

    assert sha256_payload(left) == sha256_payload(right)


def test_evidence_bundle_embeds_artifacts_and_verifies():
    baseline, candidate, report = autopilot_report()

    bundle = build_evidence_bundle(
        autopilot=report,
        baseline=baseline,
        candidate=candidate,
        provenance=provenance(),
    )
    verification = verify_bundle(bundle)

    assert verification.valid is True
    assert verification.bundle_hash_matches is True
    assert len(bundle.sha256) == 64
    assert [artifact.kind for artifact in bundle.artifacts] == [
        "autopilot",
        "optimization.baseline",
        "optimization.candidate",
    ]


def test_evidence_bundle_detects_payload_tampering():
    baseline, candidate, report = autopilot_report()
    bundle = build_evidence_bundle(
        autopilot=report,
        baseline=baseline,
        candidate=candidate,
        provenance=provenance(),
    )

    bundle.artifacts[1].payload["id"] = "tampered"
    verification = verify_bundle(bundle)

    assert verification.valid is False
    assert verification.bundle_hash_matches is False
    assert verification.artifact_failures == (
        "optimization.baseline:baseline",
    )


def test_bundle_rejects_mismatched_optimization_reference():
    _, candidate, report = autopilot_report()
    wrong_baseline = optimization("wrong", 1, 100.0)

    with pytest.raises(ValueError, match="baseline optimization"):
        build_evidence_bundle(
            autopilot=report,
            baseline=wrong_baseline,
            candidate=candidate,
            provenance=provenance(),
        )
