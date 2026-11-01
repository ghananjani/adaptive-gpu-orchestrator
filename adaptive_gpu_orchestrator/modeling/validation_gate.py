"""Explicit, read-only research acceptance gates for measured held-out errors.

Thresholds are user-specified study criteria, not universal performance SLAs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    min_platforms: int = 3
    min_test_samples_per_platform: int = 1
    max_conditioned_rmse: float | None = None
    min_fold_rmse_reduction_pct: float | None = None
    min_bootstrap_valid_fraction: float = 0.9

    def __post_init__(self) -> None:
        if self.min_platforms < 3 or self.min_test_samples_per_platform < 1:
            raise ValueError("minimum platform and per-platform sample counts are invalid")
        if not 0 <= self.min_bootstrap_valid_fraction <= 1:
            raise ValueError("bootstrap fraction must be between zero and one")
        for name in ("max_conditioned_rmse", "min_fold_rmse_reduction_pct"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.max_conditioned_rmse is not None and self.max_conditioned_rmse < 0:
            raise ValueError("maximum conditioned RMSE must be nonnegative")


def evaluate_validation_gate(report: dict[str, Any], policy: ValidationPolicy) -> dict[str, Any]:
    """Reject insufficient evidence and explicit error-threshold violations."""
    if report.get("kind") != "gpu-hardware-research-report":
        raise ValueError("invalid research report kind")
    folds = report["source"]["comparison"]["evaluation"]["folds"]
    summary = report["summary"]
    if not isinstance(folds, list) or not folds:
        raise ValueError("report has no held-out folds")
    failures = []
    if len(folds) < policy.min_platforms:
        failures.append(f"only {len(folds)} hardware platforms evaluated")
    for fold in folds:
        platform = str(fold["held_out_platform"])
        if fold["test_samples"] < policy.min_test_samples_per_platform:
            failures.append(f"{platform}: insufficient held-out samples")
        rmse = fold["conditioned_rmse"]
        if not isinstance(rmse, (int, float)) or not math.isfinite(rmse) or rmse < 0:
            raise ValueError(f"{platform}: invalid conditioned RMSE")
        if policy.max_conditioned_rmse is not None and rmse > policy.max_conditioned_rmse:
            failures.append(f"{platform}: conditioned RMSE exceeds limit")
        gain = fold["rmse_reduction_pct"]
        if policy.min_fold_rmse_reduction_pct is not None and (
            gain is None or not math.isfinite(gain)
            or gain < policy.min_fold_rmse_reduction_pct
        ):
            failures.append(f"{platform}: RMSE reduction below minimum")
    requested = summary["requested_iterations"]
    valid = summary["valid_iterations"]
    if not isinstance(requested, int) or requested < 1 or not isinstance(valid, int):
        raise ValueError("invalid bootstrap draw counts")
    if valid < 0 or valid > requested:
        raise ValueError("invalid bootstrap draw counts")
    fraction = valid / requested
    if fraction < policy.min_bootstrap_valid_fraction:
        failures.append("too few valid bootstrap refitting draws")
    return {
        "schema_version": 1,
        "kind": "research-validation-gate",
        "passed": not failures,
        "failures": failures,
        "evaluated_platforms": len(folds),
        "valid_bootstrap_fraction": fraction,
        "criteria": {
            "min_platforms": policy.min_platforms,
            "min_test_samples_per_platform": policy.min_test_samples_per_platform,
            "max_conditioned_rmse": policy.max_conditioned_rmse,
            "min_fold_rmse_reduction_pct": policy.min_fold_rmse_reduction_pct,
            "min_bootstrap_valid_fraction": policy.min_bootstrap_valid_fraction,
        },
        "limitations": "Offline evidence acceptance only; not a deployment approval or unseen-GPU calibration.",
    }
