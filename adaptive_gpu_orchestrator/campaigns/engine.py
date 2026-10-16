from __future__ import annotations

import random
from collections.abc import Callable
from uuid import uuid4

from adaptive_gpu_orchestrator.optimizer import (
    CanaryRunner,
    ConfigurationOptimizer,
    OptimizationResult,
    OptimizationStore,
)
from adaptive_gpu_orchestrator.studies import StudyStore, analyze_study

from .manifest import CampaignManifest
from .models import CampaignProvenance, CampaignReport, CampaignRun, CampaignSpec

RunnerFactory = Callable[[int, dict[str, str]], CanaryRunner]


def _candidate_order(
    spec: CampaignSpec,
    replicate: int,
):
    candidates = list(spec.candidates)
    if spec.randomize_candidates:
        if spec.random_seed is None:
            raise ValueError("randomized campaign is missing random_seed")
        generator = random.Random(spec.random_seed + replicate)
        generator.shuffle(candidates)
    return candidates


def _campaign_environment(
    report: CampaignReport,
    replicate: int,
) -> dict[str, str]:
    env = {
        "AGO_CAMPAIGN_ID": report.id,
        "AGO_CAMPAIGN_NAME": report.spec.name,
        "AGO_CAMPAIGN_GROUP": report.spec.group,
        "AGO_REPLICATE_INDEX": str(replicate),
        "AGO_CAMPAIGN_MANIFEST_SHA256": report.manifest_sha256 or "",
    }
    for key, value in report.spec.tags.items():
        normalized = "".join(
            character if character.isalnum() else "_"
            for character in key.upper()
        ).strip("_")
        if normalized:
            env[f"AGO_CAMPAIGN_TAG_{normalized}"] = value
    return env


def _execute_replicate(
    *,
    report: CampaignReport,
    replicate: int,
    runner_factory: RunnerFactory,
    optimization_store: OptimizationStore,
) -> tuple[CampaignRun, OptimizationResult | None]:
    candidates = _candidate_order(report.spec, replicate)
    candidate_order = tuple(candidate.label() for candidate in candidates)
    env = _campaign_environment(report, replicate)

    try:
        runner = runner_factory(replicate, env)
        result = ConfigurationOptimizer(
            runner=runner,
            objective=report.spec.objective,
            constraints=report.spec.constraints,
        ).optimize(
            candidates,
            baseline=report.spec.candidates[0],
        )
        optimization_store.save(result)
    except (OSError, ValueError) as exc:
        return (
            CampaignRun(
                replicate=replicate,
                status="error",
                error=str(exc),
                candidate_order=candidate_order,
            ),
            None,
        )

    if result.best is None:
        return (
            CampaignRun(
                replicate=replicate,
                status="no_feasible_candidate",
                optimization_id=result.id,
                candidate_order=candidate_order,
            ),
            result,
        )

    return (
        CampaignRun(
            replicate=replicate,
            status="completed",
            optimization_id=result.id,
            best_config=result.best.config.label(),
            candidate_order=candidate_order,
        ),
        result,
    )


def _static_provenance_key(
    provenance: CampaignProvenance,
) -> dict:
    snapshot = provenance.cluster_snapshot
    gpu_rows = []
    for item in snapshot.get("gpus", []):
        gpu_rows.append(
            {
                "name": item.get("name"),
                "memory_total_mib": item.get("memory_total_mib"),
                "compute_capability": item.get("compute_capability"),
            }
        )
    gpu_rows.sort(
        key=lambda item: (
            str(item.get("name")),
            int(item.get("memory_total_mib") or 0),
            str(item.get("compute_capability")),
        )
    )
    return {
        "driver_version": snapshot.get("driver_version"),
        "cuda_version": snapshot.get("cuda_version"),
        "gpus": gpu_rows,
        "package_versions": dict(sorted(provenance.package_versions.items())),
    }


def _refresh_study(
    report: CampaignReport,
    *,
    optimization_store: OptimizationStore,
    study_store: StudyStore,
) -> None:
    successful = [
        optimization_store.load(identifier)
        for identifier in report.completed_optimization_ids
    ]
    if successful:
        study = analyze_study({report.spec.group: successful})
        study_store.save(study)
        report.study_id = study.id
    else:
        report.study_id = None

    report.findings = []
    report.findings.append(
        f"{len(successful)}/{report.spec.repeats} replicates completed with a "
        "feasible optimizer winner"
    )
    failed = report.spec.repeats - len(successful)
    if failed:
        report.findings.append(
            f"{failed} replicate(s) remain incomplete or without a feasible winner"
        )
    if report.resume_count:
        report.findings.append(
            f"campaign execution resumed {report.resume_count} time(s)"
        )


def run_campaign(
    *,
    spec: CampaignSpec,
    provenance: CampaignProvenance,
    runner_factory: RunnerFactory,
    optimization_store: OptimizationStore,
    study_store: StudyStore,
    campaign_id: str | None = None,
) -> CampaignReport:
    manifest = CampaignManifest.create(spec)
    report = CampaignReport(
        id=campaign_id or uuid4().hex[:12],
        spec=spec,
        provenance=provenance,
        runs=[],
        manifest_sha256=manifest.sha256,
    )

    for replicate in range(1, spec.repeats + 1):
        run, _ = _execute_replicate(
            report=report,
            replicate=replicate,
            runner_factory=runner_factory,
            optimization_store=optimization_store,
        )
        report.runs.append(run)

    _refresh_study(
        report,
        optimization_store=optimization_store,
        study_store=study_store,
    )
    return report


def resume_campaign(
    report: CampaignReport,
    *,
    runner_factory: RunnerFactory,
    optimization_store: OptimizationStore,
    study_store: StudyStore,
    current_provenance: CampaignProvenance | None = None,
    allow_provenance_change: bool = False,
) -> CampaignReport:
    current_manifest = CampaignManifest.create(report.spec)
    if (
        report.manifest_sha256 is not None
        and report.manifest_sha256 != current_manifest.sha256
    ):
        raise ValueError(
            "campaign specification no longer matches its recorded manifest fingerprint"
        )
    report.manifest_sha256 = current_manifest.sha256

    pending = report.pending_replicates
    if not pending:
        raise ValueError("campaign has no incomplete replicates to resume")

    if current_provenance is not None:
        changed = (
            _static_provenance_key(current_provenance)
            != _static_provenance_key(report.provenance)
        )
        if changed and not allow_provenance_change:
            raise ValueError(
                "campaign hardware/software provenance changed; "
                "use an explicit override to resume anyway"
            )
        report.resume_provenance.append(current_provenance)

    runs = {run.replicate: run for run in report.runs}
    for replicate in pending:
        run, _ = _execute_replicate(
            report=report,
            replicate=replicate,
            runner_factory=runner_factory,
            optimization_store=optimization_store,
        )
        runs[replicate] = run

    report.runs = [
        runs[replicate]
        for replicate in sorted(runs)
    ]
    report.resume_count += 1
    _refresh_study(
        report,
        optimization_store=optimization_store,
        study_store=study_store,
    )
    return report


def compare_campaigns(
    campaigns: list[CampaignReport],
    *,
    optimization_store: OptimizationStore,
    study_store: StudyStore,
):
    if len(campaigns) < 2:
        raise ValueError("campaign comparison requires at least two campaigns")

    groups: dict[str, list[OptimizationResult]] = {}
    for campaign in campaigns:
        if campaign.spec.group in groups:
            raise ValueError(
                f"duplicate campaign group label: {campaign.spec.group}"
            )

        identifiers = campaign.completed_optimization_ids
        if not identifiers:
            raise ValueError(
                f"campaign {campaign.id} has no completed optimization runs"
            )
        groups[campaign.spec.group] = [
            optimization_store.load(identifier)
            for identifier in identifiers
        ]

    study = analyze_study(groups)
    study_store.save(study)
    return study
