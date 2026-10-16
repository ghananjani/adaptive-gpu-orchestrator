from __future__ import annotations

from uuid import uuid4
from collections.abc import Callable

from adaptive_gpu_orchestrator.optimizer import (
    CanaryRunner,
    ConfigurationOptimizer,
    OptimizationResult,
    OptimizationStore,
)
from adaptive_gpu_orchestrator.studies import StudyStore, analyze_study

from .models import (
    CampaignProvenance,
    CampaignReport,
    CampaignRun,
    CampaignSpec,
)


RunnerFactory = Callable[[int, dict[str, str]], CanaryRunner]


def run_campaign(
    *,
    spec: CampaignSpec,
    provenance: CampaignProvenance,
    runner_factory: RunnerFactory,
    optimization_store: OptimizationStore,
    study_store: StudyStore,
    campaign_id: str | None = None,
) -> CampaignReport:
    report = CampaignReport(
        id=campaign_id or uuid4().hex[:12],
        spec=spec,
        provenance=provenance,
        runs=[],
    )
    successful: list[OptimizationResult] = []

    for replicate in range(1, spec.repeats + 1):
        env = {
            "AGO_CAMPAIGN_ID": report.id,
            "AGO_CAMPAIGN_NAME": spec.name,
            "AGO_CAMPAIGN_GROUP": spec.group,
            "AGO_REPLICATE_INDEX": str(replicate),
        }
        for key, value in spec.tags.items():
            normalized = "".join(
                character if character.isalnum() else "_"
                for character in key.upper()
            ).strip("_")
            if normalized:
                env[f"AGO_CAMPAIGN_TAG_{normalized}"] = value

        try:
            runner = runner_factory(replicate, env)
            result = ConfigurationOptimizer(
                runner=runner,
                objective=spec.objective,
                constraints=spec.constraints,
            ).optimize(
                list(spec.candidates),
                baseline=spec.candidates[0],
            )
            optimization_store.save(result)
        except (OSError, ValueError) as exc:
            report.runs.append(
                CampaignRun(
                    replicate=replicate,
                    status="error",
                    error=str(exc),
                )
            )
            continue

        if result.best is None:
            report.runs.append(
                CampaignRun(
                    replicate=replicate,
                    status="no_feasible_candidate",
                    optimization_id=result.id,
                )
            )
            continue

        successful.append(result)
        report.runs.append(
            CampaignRun(
                replicate=replicate,
                status="completed",
                optimization_id=result.id,
                best_config=result.best.config.label(),
            )
        )

    if successful:
        study = analyze_study({spec.group: successful})
        study_store.save(study)
        report.study_id = study.id
        report.findings.append(
            f"{len(successful)}/{spec.repeats} replicates completed with a "
            "feasible optimizer winner"
        )
    else:
        report.findings.append(
            "no replicate produced a feasible optimizer winner; study not created"
        )

    failed = spec.repeats - len(successful)
    if failed:
        report.findings.append(
            f"{failed} replicate(s) did not produce a completed optimization"
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
