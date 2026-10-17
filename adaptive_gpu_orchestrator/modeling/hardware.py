"""Explicit measured hardware descriptors and descriptive OOD diagnostics.

No SKU specifications are guessed; callers supply verified measurements.
The diagnostics do not constitute predictive confidence intervals.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

HARDWARE_FEATURE_NAMES = (
    "gpu_count",
    "memory_gib_per_gpu",
    "memory_bandwidth_gbps_per_gpu",
    "compute_tflops_per_gpu",
    "interconnect_bandwidth_gbps",
)


@dataclass(frozen=True, slots=True)
class HardwareFeatures:
    gpu_count: int
    memory_gib_per_gpu: float
    memory_bandwidth_gbps_per_gpu: float
    compute_tflops_per_gpu: float
    interconnect_bandwidth_gbps: float

    def __post_init__(self) -> None:
        for name in HARDWARE_FEATURE_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not isinstance(self.gpu_count, int):
            raise TypeError("gpu_count must be an integer")

    def vector(self) -> tuple[float, ...]:
        return tuple(math.log2(getattr(self, name)) for name in HARDWARE_FEATURE_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in HARDWARE_FEATURE_NAMES}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HardwareFeatures:
        if any(name not in payload for name in HARDWARE_FEATURE_NAMES):
            raise ValueError("all measured hardware features are required")
        return cls(
            gpu_count=payload["gpu_count"],
            memory_gib_per_gpu=payload["memory_gib_per_gpu"],
            memory_bandwidth_gbps_per_gpu=payload["memory_bandwidth_gbps_per_gpu"],
            compute_tflops_per_gpu=payload["compute_tflops_per_gpu"],
            interconnect_bandwidth_gbps=payload["interconnect_bandwidth_gbps"],
        )


@dataclass(frozen=True, slots=True)
class HardwareOODReport:
    distance: float
    threshold: float
    out_of_distribution: bool
    outside_observed_range: tuple[str, ...]
    reference_samples: int
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "threshold": self.threshold,
            "out_of_distribution": self.out_of_distribution,
            "outside_observed_range": list(self.outside_observed_range),
            "reference_samples": self.reference_samples,
            "findings": list(self.findings),
        }


def assess_hardware_ood(
    reference: Iterable[HardwareFeatures],
    candidate: HardwareFeatures,
    *,
    threshold: float = 2.0,
) -> HardwareOODReport:
    """Compare a candidate to known, measured hardware configurations.

    Distance is nearest-reference RMS normalized log2 difference. Scale is
    estimated from reference standard deviations, with a floor of 1.0 log2
    unit to avoid division by zero and explosive single-platform scores.
    A value above threshold or outside any observed feature range is flagged.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    known = tuple(reference)
    if not known:
        raise ValueError("at least one reference hardware observation is required")
    vectors = [item.vector() for item in known]
    target = candidate.vector()
    dims = len(HARDWARE_FEATURE_NAMES)
    means = [
        math.fsum(row[i] for row in vectors) / len(vectors)
        for i in range(dims)
    ]
    scales = [
        max(
            1.0,
            math.sqrt(
                math.fsum((row[i] - means[i]) ** 2 for row in vectors)
                / len(vectors)
            ),
        )
        for i in range(dims)
    ]
    nearest = min(
        math.sqrt(
            math.fsum(
                ((target[i] - row[i]) / scales[i]) ** 2
                for i in range(dims)
            ) / dims
        )
        for row in vectors
    )
    outside = tuple(
        name for i, name in enumerate(HARDWARE_FEATURE_NAMES)
        if target[i] < min(row[i] for row in vectors)
        or target[i] > max(row[i] for row in vectors)
    )
    flagged = nearest > threshold or bool(outside)
    findings = (
        "Descriptive feature-space distance only; not calibrated uncertainty.",
        "Out-of-range features require independent benchmarking before deployment.",
        "Bandwidth and compute units must use consistent measurement conventions.",
    )
    return HardwareOODReport(
        distance=nearest,
        threshold=threshold,
        out_of_distribution=flagged,
        outside_observed_range=outside,
        reference_samples=len(known),
        findings=findings,
    )
