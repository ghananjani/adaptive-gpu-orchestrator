from __future__ import annotations

from adaptive_gpu_orchestrator.autopilot import AutopilotReport
from adaptive_gpu_orchestrator.correlation import CorrelationReport
from adaptive_gpu_orchestrator.optimizer import OptimizationResult
from adaptive_gpu_orchestrator.studies import StudyReport
from adaptive_gpu_orchestrator.timeseries import TimeSeriesReport

from .hash import sha256_payload
from .models import (
    BundleVerification,
    EvidenceArtifact,
    EvidenceBundle,
    RuntimeProvenance,
)


def _artifact(kind: str, identifier: str, payload: dict) -> EvidenceArtifact:
    return EvidenceArtifact.create(
        kind=kind,
        identifier=identifier,
        payload=payload,
    )


def build_evidence_bundle(
    *,
    autopilot: AutopilotReport,
    baseline: OptimizationResult,
    candidate: OptimizationResult,
    provenance: RuntimeProvenance,
    study: StudyReport | None = None,
    correlation: CorrelationReport | None = None,
    timeseries: TimeSeriesReport | None = None,
) -> EvidenceBundle:
    if baseline.id != autopilot.baseline_optimization_id:
        raise ValueError("baseline optimization does not match autopilot report")
    if candidate.id != autopilot.candidate_optimization_id:
        raise ValueError("candidate optimization does not match autopilot report")

    if autopilot.study_id is not None:
        if study is None or study.id != autopilot.study_id:
            raise ValueError("study evidence does not match autopilot report")
    elif study is not None:
        raise ValueError("unexpected study evidence for autopilot report")

    if autopilot.correlation_id is not None:
        if correlation is None or correlation.id != autopilot.correlation_id:
            raise ValueError("correlation evidence does not match autopilot report")
    elif correlation is not None:
        raise ValueError("unexpected correlation evidence for autopilot report")

    if autopilot.timeseries_id is not None:
        if timeseries is None or timeseries.id != autopilot.timeseries_id:
            raise ValueError("time-series evidence does not match autopilot report")
    elif timeseries is not None:
        raise ValueError("unexpected time-series evidence for autopilot report")

    artifacts = [
        _artifact("autopilot", autopilot.id, autopilot.to_dict()),
        _artifact(
            "optimization.baseline",
            baseline.id,
            baseline.to_dict(),
        ),
        _artifact(
            "optimization.candidate",
            candidate.id,
            candidate.to_dict(),
        ),
    ]

    if study is not None:
        artifacts.append(_artifact("study", study.id, study.to_dict()))
    if correlation is not None:
        artifacts.append(
            _artifact("correlation", correlation.id, correlation.to_dict())
        )
    if timeseries is not None:
        artifacts.append(
            _artifact("timeseries", timeseries.id, timeseries.to_dict())
        )

    return EvidenceBundle(
        autopilot_report_id=autopilot.id,
        artifacts=artifacts,
        provenance=provenance,
    ).seal()


def verify_bundle(bundle: EvidenceBundle) -> BundleVerification:
    failures = tuple(
        f"{artifact.kind}:{artifact.identifier}"
        for artifact in bundle.artifacts
        if sha256_payload(artifact.payload) != artifact.sha256
    )
    bundle_hash_matches = bundle.recompute_sha256() == bundle.sha256
    return BundleVerification(
        valid=not failures and bundle_hash_matches,
        bundle_hash_matches=bundle_hash_matches,
        artifact_failures=failures,
    )
