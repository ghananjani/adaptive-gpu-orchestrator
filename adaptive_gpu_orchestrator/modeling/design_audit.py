"""Read-only design audit for campaign-replicate leakage and confounding."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.campaigns import CampaignReport, campaign_trial_rows
from adaptive_gpu_orchestrator.optimizer import OptimizationStore

from .campaign_evidence import SOFTWARE_FIELDS


@dataclass(frozen=True, slots=True)
class CampaignDesignAudit:
    campaign_count: int
    platform_count: int
    per_platform_campaigns: dict[str, tuple[str, ...]]
    replicate_counts: dict[str, int]
    missing_provenance: dict[str, tuple[str, ...]]
    shared_optimization_ids: tuple[str, ...]
    platform_campaign_confounded: bool
    independent_campaign_replication: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "campaign-design-audit",
            "campaign_count": self.campaign_count,
            "platform_count": self.platform_count,
            "per_platform_campaigns": {
                k: list(v) for k, v in sorted(self.per_platform_campaigns.items())
            },
            "replicate_counts": dict(sorted(self.replicate_counts.items())),
            "missing_provenance": {
                k: list(v) for k, v in sorted(self.missing_provenance.items())
            },
            "shared_optimization_ids": list(self.shared_optimization_ids),
            "platform_campaign_confounded": self.platform_campaign_confounded,
            "independent_campaign_replication": self.independent_campaign_replication,
            "warnings": list(self.warnings),
        }


def audit_campaign_design(
    campaigns: Iterable[CampaignReport], *, optimization_store: OptimizationStore
) -> CampaignDesignAudit:
    """Audit experimental independence; do not infer replication from trial counts.

    Every campaign must have a single unambiguous GPU platform. Shared stored
    optimization IDs across campaigns are reported as possible reuse/leakage.
    """
    ordered = sorted(campaigns, key=lambda c: c.id)
    if not ordered or len({c.id for c in ordered}) != len(ordered):
        raise ValueError("campaigns must be nonempty with unique IDs")
    platform_campaigns: dict[str, set[str]] = defaultdict(set)
    replicate_counts: dict[str, int] = {}
    missing: dict[str, tuple[str, ...]] = {}
    optimization_owners: dict[str, set[str]] = defaultdict(set)
    for campaign in ordered:
        rows = campaign_trial_rows(campaign, optimization_store=optimization_store)
        if not rows:
            raise ValueError(f"campaign {campaign.id} has no saved trials")
        platforms = {str(row.get("gpu_models") or "").strip() for row in rows}
        if len(platforms) != 1 or not next(iter(platforms)):
            raise ValueError(f"ambiguous hardware platform in campaign {campaign.id}")
        platform = next(iter(platforms))
        platform_campaigns[platform].add(campaign.id)
        replicates = {str(row["replicate"]) for row in rows}
        replicate_counts[campaign.id] = len(replicates)
        missing[campaign.id] = tuple(
            field for field in SOFTWARE_FIELDS
            if any(not str(row.get(field) or "").strip() for row in rows)
        )
        for row in rows:
            optimization_id = str(row.get("optimization_id") or "").strip()
            if optimization_id:
                optimization_owners[optimization_id].add(campaign.id)
    shared = tuple(sorted(
        key for key, owners in optimization_owners.items() if len(owners) > 1
    ))
    counts = {platform: tuple(sorted(ids)) for platform, ids in sorted(platform_campaigns.items())}
    confounded = all(len(ids) == 1 for ids in counts.values())
    replicated = all(len(ids) >= 2 for ids in counts.values())
    warnings = []
    if confounded:
        warnings.append("Each platform has one campaign: campaign and hardware effects are confounded.")
    if not replicated:
        warnings.append("Not all platforms have independent campaign replication; trial repeats are not campaign replicates.")
    if shared:
        warnings.append("Stored optimization IDs are shared across campaigns; check evidence reuse/leakage.")
    if any(missing.values()):
        warnings.append("Incomplete software provenance limits comparability and reproducibility.")
    warnings.append("Descriptive design audit; no independence or causal validity is established.")
    return CampaignDesignAudit(
        campaign_count=len(ordered),
        platform_count=len(counts),
        per_platform_campaigns=counts,
        replicate_counts=replicate_counts,
        missing_provenance=missing,
        shared_optimization_ids=shared,
        platform_campaign_confounded=confounded,
        independent_campaign_replication=replicated,
        warnings=tuple(warnings),
    )
