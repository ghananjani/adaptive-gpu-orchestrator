"""Validated, read-only saved campaign bridge for refitting uncertainty."""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_gpu_orchestrator.campaigns import CampaignReport, campaign_trial_rows
from adaptive_gpu_orchestrator.optimizer import OptimizationStore, ParallelismConfig

from .campaign_evidence import CampaignEvidenceComparison, compare_saved_campaigns
from .design_audit import CampaignDesignAudit, audit_campaign_design
from .hardware import HardwareFeatures
from .refitting import (
    CampaignHardwareTrial,
    RefittingBootstrapReport,
    refit_campaign_bootstrap,
)


@dataclass(frozen=True, slots=True)
class SavedRefittingEvaluation:
    comparison: CampaignEvidenceComparison
    design_audit: CampaignDesignAudit
    refitting: RefittingBootstrapReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "saved-campaign-refitting-evaluation",
            "comparison": self.comparison.to_dict(),
            "design_audit": self.design_audit.to_dict(),
            "refitting": self.refitting.to_dict(),
        }


def evaluate_saved_refitting(
    campaigns: Iterable[CampaignReport],
    *,
    optimization_store: OptimizationStore,
    measured_hardware: Mapping[str, HardwareFeatures],
    metric: str = "throughput_per_s",
    ridge_lambda: float = 1.0,
    allow_software_mismatch: bool = False,
    iterations: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> SavedRefittingEvaluation:
    """Validate saved evidence once, then bootstrap held-out model refits.

    Campaign identity is preserved across trials; experiments are never
    executed, and out-of-distribution bypass remains evaluation-only.
    """
    selected = sorted(campaigns, key=lambda campaign: campaign.id)
    comparison = compare_saved_campaigns(
        selected,
        optimization_store=optimization_store,
        measured_hardware=measured_hardware,
        metric=metric,
        ridge_lambda=ridge_lambda,
        allow_software_mismatch=allow_software_mismatch,
    )
    audit = audit_campaign_design(selected, optimization_store=optimization_store)
    if audit.shared_optimization_ids:
        raise ValueError("shared optimization IDs across campaigns: possible evidence leakage")
    if not audit.independent_campaign_replication:
        raise ValueError("at least two separate campaigns per hardware platform required")
    trials: list[CampaignHardwareTrial] = []
    for campaign in selected:
        for row in campaign_trial_rows(campaign, optimization_store=optimization_store):
            if row.get("exit_code") != 0:
                continue
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                continue
            platform = str(row["gpu_models"]).strip()
            config = ParallelismConfig(
                tensor_parallel=int(row["tensor_parallel"]),
                pipeline_parallel=int(row["pipeline_parallel"]),
                data_parallel=int(row["data_parallel"]),
                batch_size=int(row["batch_size"]),
                concurrency=int(row["concurrency"]),
            )
            trials.append(CampaignHardwareTrial(
                platform=platform,
                campaign_id=campaign.id,
                config=config,
                hardware=measured_hardware[platform],
                target=float(value),
            ))
    if len(trials) != comparison.trial_count:
        raise ValueError("evidence count changed between validation and refitting")
    result = refit_campaign_bootstrap(
        trials,
        metric=metric,
        ridge_lambda=ridge_lambda,
        iterations=iterations,
        confidence=confidence,
        seed=seed,
    )
    if result.test_samples != comparison.evaluation.test_samples:
        raise ValueError("held-out trial count disagrees with original comparison")
    original = comparison.evaluation.rmse_reduction_pct
    if original is None or result.point_reduction_pct is None:
        if original is not None or result.point_reduction_pct is not None:
            raise ValueError("refitting point estimate diverges from comparison")
    elif not math.isclose(
        original, result.point_reduction_pct, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("refitting point estimate diverges from comparison")
    return SavedRefittingEvaluation(comparison, audit, result)


def write_saved_refitting_report(
    result: SavedRefittingEvaluation, path: str | Path
) -> Path:
    """Export deterministic JSON without changing source campaign data."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
