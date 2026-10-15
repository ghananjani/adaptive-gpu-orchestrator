from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from adaptive_gpu_orchestrator.discovery import NvidiaSMI, NvidiaSMIError
from adaptive_gpu_orchestrator.models import GPUDevice


class TelemetryProvider(Protocol):
    """Minimal interface required by continuous telemetry sampling."""

    def sample(self) -> list[GPUDevice]:
        ...


@dataclass(slots=True)
class NvidiaSMIProvider:
    """Dependency-free provider backed by nvidia-smi CSV queries."""

    nvidia_smi: NvidiaSMI

    def __init__(self, nvidia_smi: NvidiaSMI | None = None) -> None:
        self.nvidia_smi = nvidia_smi or NvidiaSMI()

    def sample(self) -> list[GPUDevice]:
        try:
            return self.nvidia_smi.query_gpus()
        except NvidiaSMIError:
            return []
