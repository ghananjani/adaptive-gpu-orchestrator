from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any
from uuid import uuid4

from adaptive_gpu_orchestrator.telemetry.sampler import TelemetrySample, average_sample


@dataclass(slots=True)
class WorkloadProfile:
    name: str
    command: list[str] = field(default_factory=list)
    samples: list[TelemetrySample] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    duration_s: float = 0.0
    exit_code: int | None = None
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    tags: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        telemetry = average_sample(self.samples)
        sample_count = len(self.samples)
        gpu_counts = [sample.gpu_count for sample in self.samples]
        return {
            "id": self.id,
            "name": self.name,
            "duration_s": self.duration_s,
            "exit_code": self.exit_code,
            "sample_count": sample_count,
            "gpu_count_max": max(gpu_counts, default=0),
            **telemetry,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "command": list(self.command),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "exit_code": self.exit_code,
            "tags": dict(self.tags),
            "samples": [sample.to_dict() for sample in self.samples],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkloadProfile":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            command=[str(item) for item in data.get("command", [])],
            started_at=str(data["started_at"]),
            finished_at=data.get("finished_at"),
            duration_s=float(data.get("duration_s", 0.0)),
            exit_code=data.get("exit_code"),
            tags={str(k): str(v) for k, v in data.get("tags", {}).items()},
            samples=[
                TelemetrySample.from_dict(sample)
                for sample in data.get("samples", [])
            ],
        )
