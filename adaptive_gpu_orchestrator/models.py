from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Bottleneck(str, Enum):
    HEALTHY = "healthy"
    NO_GPU = "no_gpu"
    UNDERUTILIZED = "underutilized"
    MEMORY_PRESSURE = "memory_pressure"
    COMMUNICATION_BOUND = "communication_bound"
    LOAD_IMBALANCE = "load_imbalance"
    THERMAL_OR_POWER = "thermal_or_power"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class GPUDevice:
    index: int
    name: str
    uuid: str = ""
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    utilization_gpu_pct: float | None = None
    utilization_memory_pct: float | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    pci_bus_id: str = ""
    compute_capability: str = ""
    mig_mode: str = ""

    @property
    def memory_pressure_pct(self) -> float | None:
        if self.memory_total_mib in (None, 0) or self.memory_used_mib is None:
            return None
        return 100.0 * self.memory_used_mib / self.memory_total_mib

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_pressure_pct"] = self.memory_pressure_pct
        return data


@dataclass(slots=True)
class TopologyLink:
    source: int
    target: int
    kind: str
    weight: int
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NCCLResult:
    operation: str
    size_bytes: int
    time_us: float
    algbw_gbps: float
    busbw_gbps: float
    out_of_place: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClusterSnapshot:
    hostname: str
    gpus: list[GPUDevice] = field(default_factory=list)
    topology: list[TopologyLink] = field(default_factory=list)
    driver_version: str | None = None
    cuda_version: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "topology": [link.to_dict() for link in self.topology],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class DiagnosisEvidence:
    signal: str
    value: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Recommendation:
    priority: str
    action: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DiagnosisReport:
    bottleneck: Bottleneck
    severity: str
    summary: str
    evidence: list[DiagnosisEvidence] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bottleneck": self.bottleneck.value,
            "severity": self.severity,
            "summary": self.summary,
            "evidence": [item.to_dict() for item in self.evidence],
            "recommendations": [item.to_dict() for item in self.recommendations],
        }
