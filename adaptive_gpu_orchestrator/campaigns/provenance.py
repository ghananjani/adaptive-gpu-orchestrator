from __future__ import annotations

import os
import re
import subprocess
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
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
)


def _package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in package_names:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _detect_cuda_toolkit_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)*)", completed.stdout)
    return match.group(1) if match else None


def _detect_nccl_version(package_versions: dict[str, str]) -> str | None:
    explicit = os.environ.get("NCCL_VERSION")
    if explicit:
        return explicit
    for package in ("nvidia-nccl-cu13", "nvidia-nccl-cu12", "nccl"):
        value = package_versions.get(package)
        if value:
            return value
    return None



def capture_campaign_provenance(
    *,
    tags: dict[str, str] | None = None,
    snapshot: ClusterSnapshot | None = None,
    nvidia_smi: NvidiaSMI | None = None,
    package_names: tuple[str, ...] = _DEFAULT_PACKAGES,
    cuda_toolkit_version: str | None = None,
    nccl_version: str | None = None,
    container_image: str | None = None,
    container_image_digest: str | None = None,
) -> CampaignProvenance:
    cluster = snapshot or (nvidia_smi or NvidiaSMI()).snapshot()
    package_versions = _package_versions(package_names)
    return CampaignProvenance(
        runtime=capture_provenance(tags=tags),
        cluster_snapshot=cluster.to_dict(),
        package_versions=package_versions,
        cuda_toolkit_version=(
            cuda_toolkit_version or _detect_cuda_toolkit_version()
        ),
        nccl_version=nccl_version or _detect_nccl_version(package_versions),
        container_image=(
            container_image or os.environ.get("AGO_CONTAINER_IMAGE")
        ),
        container_image_digest=(
            container_image_digest
            or os.environ.get("AGO_CONTAINER_IMAGE_DIGEST")
        ),
    )
