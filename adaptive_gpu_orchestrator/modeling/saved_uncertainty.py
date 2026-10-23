"""Read-only out-of-fold residual extraction from validated saved campaigns."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.campaigns import CampaignReport, campaign_trial_rows
from adaptive_gpu_orchestrator.optimizer import OptimizationStore, ParallelismConfig

from .bootstrap import (
    CampaignBootstrapReport,
    CampaignPairedResidual,
    bootstrap_campaign_rmse_reduction,
)
from .campaign_evidence import CampaignEvidenceComparison, compare_saved_campaigns
from .conditioned import HardwareTrainingSample, fit_hardware_conditioned_model
from .hardware import HardwareFeatures
from .linear import fit_metric_model


@dataclass(frozen=True, slots=True)
class SavedCampaignUncertainty:
    comparison: CampaignEvidenceComparison
    bootstrap: CampaignBootstrapReport
    residuals: tuple[CampaignPairedResidual, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "saved-campaign-heldout-uncertainty",
            "comparison": self.comparison.to_dict(),
            "bootstrap": self.bootstrap.to_dict(),
            "residuals": [
                {
                    "platform": r.platform,
                    "campaign_id": r.campaign_id,
                    "config_error": r.config_error,
                    "conditioned_error": r.conditioned_error,
                }
                for r in self.residuals
            ],
        }


def evaluate_saved_campaign_uncertainty(
    campaigns: Iterable[CampaignReport],
    *,
    optimization_store: OptimizationStore,
    measured_hardware: Mapping[str, HardwareFeatures],
    metric: str = "throughput_per_s",
    ridge_lambda: float = 1.0,
    allow_software_mismatch: bool = False,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> SavedCampaignUncertainty:
    """Evaluate paired leave-one-platform-out residuals then bootstrap campaigns.

    Reuses the strict saved-campaign provenance/workload validation. A held-out
    platform is extrapolation by definition, permitted here only for offline
    comparison, never for production predictions or scheduling.
    """
    selected = sorted(campaigns, key=lambda c: c.id)
    comparison = compare_saved_campaigns(
        selected,
        optimization_store=optimization_store,
        measured_hardware=measured_hardware,
        metric=metric,
        ridge_lambda=ridge_lambda,
        allow_software_mismatch=allow_software_mismatch,
    )
    grouped: dict[str, list[tuple[str, ParallelismConfig, HardwareFeatures, float]]] = defaultdict(list)
    for campaign in selected:
        for row in campaign_trial_rows(campaign, optimization_store=optimization_store):
            if row.get("exit_code") != 0:
                continue
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            platform = str(row["gpu_models"]).strip()
            config = ParallelismConfig(
                tensor_parallel=int(row["tensor_parallel"]),
                pipeline_parallel=int(row["pipeline_parallel"]),
                data_parallel=int(row["data_parallel"]),
                batch_size=int(row["batch_size"]),
                concurrency=int(row["concurrency"]),
            )
            grouped[platform].append(
                (campaign.id, config, measured_hardware[platform], float(value))
            )
    for rows in grouped.values():
        rows.sort(key=lambda r: (r[1].label(), r[3], r[0]))
    residuals = []
    for held_out in sorted(grouped):
        train = [r for platform in sorted(grouped) if platform != held_out for r in grouped[platform]]
        config_model = fit_metric_model(
            metric=metric,
            configs=[r[1] for r in train],
            targets=[r[3] for r in train],
            ridge_lambda=ridge_lambda,
        )
        conditioned_model = fit_hardware_conditioned_model(
            [HardwareTrainingSample(r[1], r[2], r[3]) for r in train],
            metric=metric,
            ridge_lambda=ridge_lambda,
        )
        for campaign_id, config, hardware, actual in grouped[held_out]:
            residuals.append(CampaignPairedResidual(
                platform=held_out,
                campaign_id=campaign_id,
                config_error=config_model.predict(config) - actual,
                conditioned_error=conditioned_model.predict(
                    config, hardware, allow_ood=True
                ) - actual,
            ))
    bootstrap = bootstrap_campaign_rmse_reduction(
        residuals, iterations=iterations, confidence=confidence, seed=seed,
    )
    # Sanity check that the newly exposed residuals agree with paired folds.
    for fold in comparison.evaluation.folds:
        selected_errors = [r for r in residuals if r.platform == fold.held_out_platform]
        n = len(selected_errors)
        base_rmse = math.sqrt(math.fsum(r.config_error ** 2 for r in selected_errors) / n)
        conditioned_rmse = math.sqrt(
            math.fsum(r.conditioned_error ** 2 for r in selected_errors) / n
        )
        if not (
            math.isclose(base_rmse, fold.config_rmse, rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(conditioned_rmse, fold.conditioned_rmse, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError("residuals diverge from paired fold evaluation")
    return SavedCampaignUncertainty(
        comparison=comparison,
        bootstrap=bootstrap,
        residuals=tuple(residuals),
    )
