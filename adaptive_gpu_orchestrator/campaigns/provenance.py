from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from adaptive_gpu_orchestrator.discovery import NvidiaSMI
from adaptive_gpu_orchestrator.evidence import capture_provenance
from adaptive_gpu_orchestrator.models import ClusterSnapshot

from .models import CampaignProvenance

_DEFAULT_PACKAGES = (
    "torch",
    "vllm",
    "triton",
    "nvidia-ml-py",
    "numpy",
)


def _package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in package_names:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def capture_campaign_provenance(
    *,
    tags: dict[str, str] | None = None,
    snapshot: ClusterSnapshot | None = None,
    nvidia_smi: NvidiaSMI | None = None,
    package_names: tuple[str, ...] = _DEFAULT_PACKAGES,
) -> CampaignProvenance:
    cluster = snapshot or (nvidia_smi or NvidiaSMI()).snapshot()
    return CampaignProvenance(
        runtime=capture_provenance(tags=tags),
        cluster_snapshot=cluster.to_dict(),
        package_versions=_package_versions(package_names),
    )
