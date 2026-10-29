"""Fail-closed metadata preflight for publication-oriented GPU campaign reports."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from adaptive_gpu_orchestrator.campaigns import CampaignReport, campaign_trial_rows
from adaptive_gpu_orchestrator.optimizer import OptimizationStore

from .campaign_evidence import SOFTWARE_FIELDS
from .hardware import HardwareFeatures


def validate_research_provenance(
    campaigns: Iterable[CampaignReport],
    *,
    optimization_store: OptimizationStore,
    measured_hardware: Mapping[str, HardwareFeatures],
) -> None:
    """Reject incomplete or internally inconsistent source metadata.

    This does not establish that user-supplied specifications are authentic.
    Runs only on saved data; no host inspection or GPU execution.
    """
    seen_optimizations: dict[str, str] = {}
    for campaign in sorted(campaigns, key=lambda item: item.id):
        rows = campaign_trial_rows(campaign, optimization_store=optimization_store)
        if not rows:
            raise ValueError(f"campaign {campaign.id} has no trial evidence")
        platforms: set[str] = set()
        identities: set[tuple[str, str, str]] = set()
        for row in rows:
            platform = str(row.get("gpu_models") or "").strip()
            if not platform or platform not in measured_hardware:
                raise ValueError(f"campaign {campaign.id} lacks measured hardware metadata")
            platforms.add(platform)
            count = row.get("gpu_count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"campaign {campaign.id} missing valid GPU count")
            if count != measured_hardware[platform].gpu_count:
                raise ValueError(f"campaign {campaign.id} GPU count mismatch")
            for field in SOFTWARE_FIELDS:
                if not str(row.get(field) or "").strip():
                    raise ValueError(f"campaign {campaign.id} missing software provenance: {field}")
            optimization = str(row.get("optimization_id") or "").strip()
            if not optimization:
                raise ValueError(f"campaign {campaign.id} missing optimization ID")
            owner = seen_optimizations.setdefault(optimization, campaign.id)
            if owner != campaign.id:
                raise ValueError(f"optimization ID {optimization} reused by campaigns {owner} and {campaign.id}")
            identity = (
                str(row.get("replicate") or "").strip(),
                optimization,
                str(row.get("trial_index") or "").strip(),
            )
            if not all(identity) or identity in identities:
                raise ValueError(f"campaign {campaign.id} duplicated or missing trial identity")
            identities.add(identity)
        if len(platforms) != 1:
            raise ValueError(f"campaign {campaign.id} mixes hardware platforms")
