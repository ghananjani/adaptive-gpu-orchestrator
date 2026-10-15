from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class PhaseWindow:
    name: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name cannot be empty")
        if self.start_s < 0:
            raise ValueError("phase start must be non-negative")
        if self.end_s <= self.start_s:
            raise ValueError("phase end must be greater than phase start")

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GPUTimeSeriesSummary:
    gpu_index: int
    sample_count: int
    gpu_util_mean_pct: float | None = None
    gpu_util_min_pct: float | None = None
    gpu_util_max_pct: float | None = None
    memory_pressure_mean_pct: float | None = None
    memory_pressure_max_pct: float | None = None
    power_draw_mean_w: float | None = None
    temperature_max_c: float | None = None
    straggler_gap_pct_points: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhaseSummary:
    name: str
    start_s: float
    end_s: float
    sample_count: int
    gpu_util_mean_pct: float | None = None
    memory_pressure_mean_pct: float | None = None
    power_draw_mean_w: float | None = None
    gpu_util_by_index: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gpu_util_by_index"] = {
            str(key): value
            for key, value in self.gpu_util_by_index.items()
        }
        return data


@dataclass(slots=True)
class TimeSeriesReport:
    profile_id: str
    gpu_summaries: list[GPUTimeSeriesSummary] = field(default_factory=list)
    straggler_gpu_indices: list[int] = field(default_factory=list)
    phases: list[PhaseSummary] = field(default_factory=list)
    dcgm_summary: dict[int, dict[str, float]] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "gpu_summaries": [item.to_dict() for item in self.gpu_summaries],
            "straggler_gpu_indices": list(self.straggler_gpu_indices),
            "phases": [phase.to_dict() for phase in self.phases],
            "dcgm_summary": {
                str(index): dict(metrics)
                for index, metrics in self.dcgm_summary.items()
            },
            "findings": list(self.findings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeSeriesReport:
        return cls(
            id=str(data["id"]),
            profile_id=str(data["profile_id"]),
            gpu_summaries=[
                GPUTimeSeriesSummary(**item)
                for item in data.get("gpu_summaries", [])
            ],
            straggler_gpu_indices=[
                int(item)
                for item in data.get("straggler_gpu_indices", [])
            ],
            phases=[
                PhaseSummary(
                    name=str(item["name"]),
                    start_s=float(item["start_s"]),
                    end_s=float(item["end_s"]),
                    sample_count=int(item["sample_count"]),
                    gpu_util_mean_pct=item.get("gpu_util_mean_pct"),
                    memory_pressure_mean_pct=item.get("memory_pressure_mean_pct"),
                    power_draw_mean_w=item.get("power_draw_mean_w"),
                    gpu_util_by_index={
                        int(key): float(value)
                        for key, value in item.get("gpu_util_by_index", {}).items()
                    },
                )
                for item in data.get("phases", [])
            ],
            dcgm_summary={
                int(index): {
                    str(name): float(value)
                    for name, value in metrics.items()
                }
                for index, metrics in data.get("dcgm_summary", {}).items()
            },
            findings=[str(item) for item in data.get("findings", [])],
        )
