from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_gpu_orchestrator.benchmarks.nccl import NCCLResult
from adaptive_gpu_orchestrator.diagnosis import diagnose
from adaptive_gpu_orchestrator.discovery import NvidiaSMI
from adaptive_gpu_orchestrator.models import ClusterSnapshot, DiagnosisReport


@dataclass(slots=True)
class Doctor:
    nvidia_smi: NvidiaSMI = field(default_factory=NvidiaSMI)

    def inspect(self) -> ClusterSnapshot:
        return self.nvidia_smi.snapshot()

    def diagnose(self, nccl: list[NCCLResult] | None = None) -> DiagnosisReport:
        return diagnose(self.inspect(), nccl=nccl)
