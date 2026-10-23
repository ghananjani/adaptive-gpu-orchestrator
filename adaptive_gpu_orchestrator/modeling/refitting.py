"""Refitting-based sensitivity analysis of paired leave-platform-out predictions.

Every draw resamples whole *training* campaigns, refits both estimators,
and scores unchanged held-out platforms. No GPU workloads are launched.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

from .conditioned import HardwareTrainingSample, fit_hardware_conditioned_model
from .hardware import HardwareFeatures
from .linear import fit_metric_model


@dataclass(frozen=True, slots=True)
class CampaignHardwareTrial:
    platform: str
    campaign_id: str
    config: ParallelismConfig
    hardware: HardwareFeatures
    target: float

    def __post_init__(self) -> None:
        if not self.platform.strip() or not self.campaign_id.strip():
            raise ValueError("platform and campaign ID must be nonempty")
        if isinstance(self.target, bool) or not isinstance(self.target, (float, int)):
            raise TypeError("target must be numeric")
        if not math.isfinite(self.target):
            raise ValueError("target must be finite")


@dataclass(frozen=True, slots=True)
class RefittingBootstrapReport:
    metric: str
    point_reduction_pct: float | None
    lower_pct: float | None
    upper_pct: float | None
    confidence: float
    iterations: int
    valid_iterations: int
    seed: int
    platform_count: int
    campaign_count: int
    test_samples: int
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "refitting-campaign-bootstrap",
            "metric": self.metric,
            "point_reduction_pct": self.point_reduction_pct,
            "lower_pct": self.lower_pct,
            "upper_pct": self.upper_pct,
            "confidence": self.confidence,
            "iterations": self.iterations,
            "valid_iterations": self.valid_iterations,
            "seed": self.seed,
            "platform_count": self.platform_count,
            "campaign_count": self.campaign_count,
            "test_samples": self.test_samples,
            "findings": list(self.findings),
        }


def _percentile(ordered: list[float], fraction: float) -> float:
    location = (len(ordered) - 1) * fraction
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def _fit_and_score(
    groups: dict[str, list[list[CampaignHardwareTrial]]],
    rng: random.Random | None,
    *,
    metric: str,
    ridge_lambda: float,
) -> tuple[float | None, int]:
    config_ss = conditioned_ss = 0.0
    n = 0
    for held_out in sorted(groups):
        train: list[CampaignHardwareTrial] = []
        for platform in sorted(groups):
            if platform == held_out:
                continue
            clusters = groups[platform]
            selected = (
                [clusters[rng.randrange(len(clusters))] for _ in clusters]
                if rng is not None else clusters
            )
            train.extend(trial for cluster in selected for trial in cluster)
        if len({trial.config for trial in train}) < 2 or len(train) < 4:
            raise ValueError("resampled training fold lacks configuration diversity")
        baseline = fit_metric_model(
            metric=metric,
            configs=[trial.config for trial in train],
            targets=[trial.target for trial in train],
            ridge_lambda=ridge_lambda,
        )
        conditioned = fit_hardware_conditioned_model(
            [HardwareTrainingSample(trial.config, trial.hardware, trial.target)
             for trial in train],
            metric=metric,
            ridge_lambda=ridge_lambda,
        )
        for cluster in groups[held_out]:
            for trial in cluster:
                config_ss += (baseline.predict(trial.config) - trial.target) ** 2
                conditioned_ss += (
                    conditioned.predict(trial.config, trial.hardware, allow_ood=True)
                    - trial.target
                ) ** 2
                n += 1
    return (None if config_ss == 0 else
            100.0 * (1.0 - math.sqrt(conditioned_ss / config_ss))), n


def refit_campaign_bootstrap(
    trials: Iterable[CampaignHardwareTrial],
    *,
    metric: str = "throughput_per_s",
    ridge_lambda: float = 1.0,
    iterations: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
) -> RefittingBootstrapReport:
    """Resample whole training campaigns within each platform, refitting each draw.

    Held-out platform measurements are never selected into its training fold.
    Held-out test campaigns are fixed: intervals reflect training-campaign
    variation conditional on observed platforms, NOT new-platform uncertainty.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 20:
        raise ValueError("iterations must be an integer >= 20")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be finite and positive")
    grouped: dict[str, dict[str, list[CampaignHardwareTrial]]] = defaultdict(
        lambda: defaultdict(list)
    )
    owners: dict[str, str] = {}
    hardware_by_platform: dict[str, HardwareFeatures] = {}
    for trial in trials:
        owner = owners.setdefault(trial.campaign_id, trial.platform)
        if owner != trial.platform:
            raise ValueError("campaign ID spans multiple hardware platforms")
        hardware = hardware_by_platform.setdefault(trial.platform, trial.hardware)
        if hardware != trial.hardware:
            raise ValueError("platform contains inconsistent hardware features")
        grouped[trial.platform][trial.campaign_id].append(trial)
    if len(grouped) < 3 or len(set(hardware_by_platform.values())) < 3:
        raise ValueError("at least three distinct hardware platforms are required")
    if any(len(campaigns) < 2 for campaigns in grouped.values()):
        raise ValueError("at least two campaigns per platform are required")
    clusters = {
        platform: [
            sorted(campaigns[identifier],
                   key=lambda row: (row.config.label(), row.target))
            for identifier in sorted(campaigns)
        ]
        for platform, campaigns in sorted(grouped.items())
    }
    point, total = _fit_and_score(clusters, None, metric=metric, ridge_lambda=ridge_lambda)
    rng = random.Random(seed)
    values = []
    skipped = 0
    for _ in range(iterations):
        try:
            value, _ = _fit_and_score(
                clusters, rng, metric=metric, ridge_lambda=ridge_lambda
            )
        except ValueError as exc:
            if "configuration diversity" not in str(exc):
                raise
            skipped += 1
            continue
        if value is not None:
            values.append(value)
    values.sort()
    fraction = (1 - confidence) / 2
    return RefittingBootstrapReport(
        metric=metric,
        point_reduction_pct=point,
        lower_pct=_percentile(values, fraction) if values else None,
        upper_pct=_percentile(values, 1 - fraction) if values else None,
        confidence=confidence,
        iterations=iterations,
        valid_iterations=len(values),
        seed=seed,
        platform_count=len(clusters),
        campaign_count=len(owners),
        test_samples=total,
        findings=(
            "Each draw refits both models on resampled non-held-out campaigns.",
            "Held-out platforms and their test observations remain fixed across draws.",
            "Intervals are conditional sensitivity estimates, not calibrated unseen-platform uncertainty.",
            f"{skipped} draws lacked configuration diversity and were excluded.",
            "Research-only OOD bypass; no scheduler or production deployment action.",
        ),
    )
