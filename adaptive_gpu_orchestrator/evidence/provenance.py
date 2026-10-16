from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

from .models import RuntimeProvenance

_DEFAULT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "OMP_NUM_THREADS",
)


def _package_version() -> str | None:
    try:
        return version("adaptive-gpu-orchestrator")
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def capture_provenance(
    *,
    tags: dict[str, str] | None = None,
    env_keys: tuple[str, ...] = _DEFAULT_ENV_KEYS,
) -> RuntimeProvenance:
    environment = {
        key: value
        for key in env_keys
        if (value := os.environ.get(key)) is not None
    }
    return RuntimeProvenance(
        python_version=platform.python_version(),
        implementation=sys.implementation.name,
        platform=platform.platform(),
        machine=platform.machine(),
        package_version=_package_version(),
        git_commit=_git_commit(),
        environment=environment,
        tags=dict(tags or {}),
    )
