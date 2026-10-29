"""Deterministic source-evidence manifests for offline GPU research.

Hashes canonical serialized content, rather than timestamps or arbitrary paths.
This detects changes but does not authenticate the original measurements.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from adaptive_gpu_orchestrator.campaigns import CampaignReport
from adaptive_gpu_orchestrator.optimizer import OptimizationStore


def _digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fingerprint_saved_evidence(
    campaigns: Iterable[CampaignReport], *, optimization_store: OptimizationStore
) -> dict[str, Any]:
    """Fingerprint full campaign and referenced optimization serialization."""
    selected = sorted(campaigns, key=lambda c: c.id)
    if not selected or len({c.id for c in selected}) != len(selected):
        raise ValueError("campaigns must be nonempty and unique")
    sources = []
    ownership: dict[str, str] = {}
    for campaign in selected:
        optimization_hashes = []
        for optimization_id in sorted(set(campaign.completed_optimization_ids)):
            owner = ownership.setdefault(optimization_id, campaign.id)
            if owner != campaign.id:
                raise ValueError("shared optimization ID across campaigns")
            optimization = optimization_store.load(optimization_id)
            if optimization.id != optimization_id:
                raise ValueError(f"optimization identity mismatch: {optimization_id}")
            optimization_hashes.append({
                "id": optimization_id,
                "sha256": _digest(optimization.to_dict()),
            })
        sources.append({
            "campaign_id": campaign.id,
            "campaign_sha256": _digest(campaign.to_dict()),
            "optimizations": optimization_hashes,
        })
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8",
        "sources": sources,
        "manifest_sha256": _digest(sources),
    }
