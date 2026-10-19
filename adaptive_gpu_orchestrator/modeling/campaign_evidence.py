"""Read-only bridge from persisted campaign evidence to paired hardware evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_gpu_orchestrator.campaigns import CampaignReport, campaign_trial_rows
from adaptive_gpu_orchestrator.optimizer import OptimizationStore, ParallelismConfig

from .comparison import (
    LabeledHardwareTrial,
    PairedHardwareEvaluation,
    compare_hardware_heldout_models,
)
from .hardware import HardwareFeatures

# Explicit equality prevents silently comparing different model, prompt, dtype,
# generation, or benchmark protocol. Operators must supply these campaign tags.
REQUIRED_TAGS = (
    "model",
    "workload",
    "precision",
    "input_tokens",
    "output_tokens",
    "benchmark_protocol",
)
SOFTWARE_FIELDS = (
    "cuda_runtime_version",
    "cuda_toolkit_version",
    "nccl_version",
    "container_image_digest",
)


@dataclass(frozen=True, slots=True)
class CampaignEvidenceComparison:
    evaluation: PairedHardwareEvaluation
    campaign_ids: tuple[str, ...]
    hardware_by_platform: dict[str, HardwareFeatures]
    shared_tags: dict[str, str]
    software_by_platform: dict[str, dict[str, str]]
    excluded_failed: int
    excluded_missing: int
    trial_count: int
    source_sha256: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "paired-hardware-campaign-evidence",
            "evaluation": self.evaluation.to_dict(),
            "campaign_ids": list(self.campaign_ids),
            "hardware_by_platform": {
                key: value.to_dict() for key, value in sorted(self.hardware_by_platform.items())
            },
            "shared_tags": dict(sorted(self.shared_tags.items())),
            "software_by_platform": {
                key: value for key, value in sorted(self.software_by_platform.items())
            },
            "excluded_failed": self.excluded_failed,
            "excluded_missing": self.excluded_missing,
            "trial_count": self.trial_count,
            "source_sha256": self.source_sha256,
            "warnings": list(self.warnings),
        }


def compare_saved_campaigns(
    campaigns: Iterable[CampaignReport],
    *,
    optimization_store: OptimizationStore,
    measured_hardware: Mapping[str, HardwareFeatures],
    metric: str = "throughput_per_s",
    ridge_lambda: float = 1.0,
    allow_software_mismatch: bool = False,
) -> CampaignEvidenceComparison:
    """Validate provenance and compare campaigns without executing GPU workloads.

    Hardware descriptors must be independently measured, keyed by exact GPU
    model signature as emitted in the saved trial exports.
    """
    selected = sorted(campaigns, key=lambda campaign: campaign.id)
    if not selected or len({c.id for c in selected}) != len(selected):
        raise ValueError("campaigns must be nonempty and have unique IDs")
    expected: dict[str, str] | None = None
    platform_software: dict[str, dict[str, str]] = {}
    platforms: dict[str, HardwareFeatures] = {}
    included: list[LabeledHardwareTrial] = []
    normalized_rows: list[dict[str, Any]] = []
    failed = missing = 0
    for campaign in selected:
        tags = campaign.spec.tags
        for key in REQUIRED_TAGS:
            if not tags.get(key, "").strip():
                raise ValueError(f"campaign {campaign.id} missing required tag: {key}")
        current = {key: tags[key] for key in REQUIRED_TAGS}
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(f"workload/protocol tags mismatch for campaign {campaign.id}")
        rows = campaign_trial_rows(campaign, optimization_store=optimization_store)
        if not rows:
            raise ValueError(f"campaign {campaign.id} has no stored trial rows")
        for row in rows:
            platform = str(row.get("gpu_models") or "").strip()
            if not platform or platform not in measured_hardware:
                raise ValueError(f"missing measured hardware for platform {platform!r}")
            features = measured_hardware[platform]
            observed_count = row.get("gpu_count")
            if observed_count is not None and int(observed_count) != features.gpu_count:
                raise ValueError(f"GPU count mismatch for {platform}")
            software = {
                key: str(row.get(key) or "") for key in SOFTWARE_FIELDS
            }
            previous = platform_software.setdefault(platform, software)
            if previous != software:
                raise ValueError(f"conflicting software provenance within platform {platform}")
            platforms[platform] = features
            if row.get("exit_code") != 0:
                failed += 1
                continue
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                missing += 1
                continue
            config = ParallelismConfig(
                tensor_parallel=int(row["tensor_parallel"]),
                pipeline_parallel=int(row["pipeline_parallel"]),
                data_parallel=int(row["data_parallel"]),
                batch_size=int(row["batch_size"]),
                concurrency=int(row["concurrency"]),
            )
            included.append(LabeledHardwareTrial(platform, config, features, float(value)))
            normalized_rows.append({
                "campaign_id": campaign.id,
                "replicate": row["replicate"],
                "trial_index": row["trial_index"],
                "platform": platform,
                "config": config.to_dict(),
                "value": float(value),
            })
    software_values = list(platform_software.values())
    inconsistent = len({tuple(sorted(item.items())) for item in software_values}) > 1
    if inconsistent and not allow_software_mismatch:
        raise ValueError("software provenance differs across platforms; explicit override required")
    if len({item.hardware for item in included}) < 3:
        raise ValueError("comparison requires three distinct measured hardware descriptors")
    evaluation = compare_hardware_heldout_models(
        included, metric=metric, ridge_lambda=ridge_lambda
    )
    canonical = json.dumps(sorted(
        normalized_rows, key=lambda row: (
            row["campaign_id"], row["replicate"], row["trial_index"],
            row["platform"], json.dumps(row["config"], sort_keys=True), row["value"]
        )
    ), sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    warnings = (
        "Software mismatch explicitly overridden; interpret platform errors cautiously.",
    ) if inconsistent else ()
    return CampaignEvidenceComparison(
        evaluation=evaluation,
        campaign_ids=tuple(item.id for item in selected),
        hardware_by_platform=platforms,
        shared_tags=expected or {},
        software_by_platform=platform_software,
        excluded_failed=failed,
        excluded_missing=missing,
        trial_count=len(included),
        source_sha256=digest,
        warnings=warnings,
    )


def write_campaign_comparison(
    result: CampaignEvidenceComparison,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic analysis artifacts. Does not save or run any campaign."""
    json_file, csv_file = Path(json_path), Path(csv_path)
    if json_file.resolve() == csv_file.resolve():
        raise ValueError("JSON and CSV output paths must differ")
    json_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(
        json.dumps(result.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = (
        "held_out_platform", "train_samples", "test_samples", "config_rmse",
        "conditioned_rmse", "config_mae", "conditioned_mae", "rmse_reduction_pct",
    )
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for fold in result.evaluation.folds:
            writer.writerow(fold.to_dict())
    return json_file, csv_file
