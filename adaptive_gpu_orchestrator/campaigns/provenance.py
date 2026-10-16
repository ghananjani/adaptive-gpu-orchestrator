from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
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
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu13",
)

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
_CUDA_RELEASE = re.compile(r"release\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


def _package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in package_names:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=3,
    )


def probe_runtime_versions(
    *,
    package_versions: dict[str, str] | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, str]:
    packages = dict(package_versions or {})
    versions: dict[str, str] = {}

    runner = command_runner or _run_command
    try:
        completed = runner(["nvcc", "--version"])
    except (OSError, subprocess.TimeoutExpired):
        completed = None

    if completed is not None and completed.returncode == 0:
        match = _CUDA_RELEASE.search(
            (completed.stdout or "") + "\n" + (completed.stderr or "")
        )
        if match:
            versions["cuda_toolkit"] = match.group(1)

    for package in ("nvidia-nccl-cu13", "nvidia-nccl-cu12"):
        if package in packages:
            versions["nccl_package"] = packages[package]
            versions["nccl_package_name"] = package
            break

    for package in (
        "nvidia-cuda-runtime-cu13",
        "nvidia-cuda-runtime-cu12",
    ):
        if package in packages:
            versions["cuda_runtime_package"] = packages[package]
            versions["cuda_runtime_package_name"] = package
            break

    return versions


def capture_campaign_provenance(
    *,
    tags: dict[str, str] | None = None,
    snapshot: ClusterSnapshot | None = None,
    nvidia_smi: NvidiaSMI | None = None,
    package_names: tuple[str, ...] = _DEFAULT_PACKAGES,
    command_runner: CommandRunner | None = None,
) -> CampaignProvenance:
    cluster = snapshot or (nvidia_smi or NvidiaSMI()).snapshot()
    packages = _package_versions(package_names)
    return CampaignProvenance(
        runtime=capture_provenance(tags=tags),
        cluster_snapshot=cluster.to_dict(),
        package_versions=packages,
        runtime_versions=probe_runtime_versions(
            package_versions=packages,
            command_runner=command_runner,
        ),
    )
