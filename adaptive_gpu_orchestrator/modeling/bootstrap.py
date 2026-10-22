"""Seeded campaign-cluster bootstrap for paired held-out residuals.

Resamples whole independent campaigns, never individual correlated requests.
Only summarizes out-of-fold predictions supplied by an upstream evaluator.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CampaignPairedResidual:
    platform: str
    campaign_id: str
    config_error: float
    conditioned_error: float

    def __post_init__(self) -> None:
        if not self.platform.strip() or not self.campaign_id.strip():
            raise ValueError("platform and campaign_id must be nonempty")
        for name in ("config_error", "conditioned_error"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class CampaignBootstrapReport:
    estimate_pct: float | None
    lower_pct: float | None
    upper_pct: float | None
    confidence: float
    iterations: int
    valid_iterations: int
    campaign_count: int
    platform_count: int
    sample_count: int
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "campaign-cluster-bootstrap",
            "estimate_pct": self.estimate_pct,
            "lower_pct": self.lower_pct,
            "upper_pct": self.upper_pct,
            "confidence": self.confidence,
            "iterations": self.iterations,
            "valid_iterations": self.valid_iterations,
            "campaign_count": self.campaign_count,
            "platform_count": self.platform_count,
            "sample_count": self.sample_count,
            "findings": list(self.findings),
        }


def _reduction(rows: list[CampaignPairedResidual]) -> float | None:
    base = math.fsum(row.config_error ** 2 for row in rows)
    adjusted = math.fsum(row.conditioned_error ** 2 for row in rows)
    if base == 0:
        return None
    return 100.0 * (1.0 - math.sqrt(adjusted / base))


def _percentile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_campaign_rmse_reduction(
    residuals: Iterable[CampaignPairedResidual],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CampaignBootstrapReport:
    """Platform-stratified, campaign-cluster percentile bootstrap.

    Each campaign's full set of paired, out-of-fold residuals stays intact.
    At least two campaigns per platform are required. No new models are fit:
    this measures variation of *fixed* out-of-fold predictions only, and
    therefore understates uncertainty from refitting and platform sampling.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 100:
        raise ValueError("iterations must be an integer >= 100")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    grouped: dict[str, dict[str, list[CampaignPairedResidual]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observations = list(residuals)
    if not observations:
        raise ValueError("at least one paired residual is required")
    owners: dict[str, str] = {}
    for row in observations:
        previous = owners.setdefault(row.campaign_id, row.platform)
        if previous != row.platform:
            raise ValueError("campaign_id appears on more than one platform")
        grouped[row.platform][row.campaign_id].append(row)
    if len(grouped) < 2:
        raise ValueError("at least two platforms are required")
    if any(len(campaigns) < 2 for campaigns in grouped.values()):
        raise ValueError("at least two independent campaigns per platform are required")
    clusters = []
    for platform in sorted(grouped):
        groups = []
        for campaign in sorted(grouped[platform]):
            groups.append(sorted(
                grouped[platform][campaign],
                key=lambda r: (r.config_error, r.conditioned_error),
            ))
        clusters.append(groups)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        sampled = [
            row
            for platform_groups in clusters
            for _ in range(len(platform_groups))
            for row in platform_groups[rng.randrange(len(platform_groups))]
        ]
        value = _reduction(sampled)
        if value is not None:
            draws.append(value)
    draws.sort()
    low = high = None
    if draws:
        tail = (1.0 - confidence) / 2
        low = _percentile(draws, tail)
        high = _percentile(draws, 1.0 - tail)
    return CampaignBootstrapReport(
        estimate_pct=_reduction(observations),
        lower_pct=low,
        upper_pct=high,
        confidence=confidence,
        iterations=iterations,
        valid_iterations=len(draws),
        campaign_count=len(owners),
        platform_count=len(grouped),
        sample_count=len(observations),
        findings=(
            "Campaigns are resampled within platforms; requests in one campaign are not independent.",
            "Percentile interval is descriptive conditional on the supplied out-of-fold predictions.",
            "Fixed predictions are not refit and held-out hardware platforms are not resampled.",
            "A positive RMSE reduction is not proof of generalization or causal hardware benefit.",
        ),
    )
