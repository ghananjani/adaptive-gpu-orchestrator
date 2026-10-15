from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig, TrialMetrics


@dataclass(slots=True)
class CorrelationReport:
    optimization_id: str
    profile_id: str
    config: ParallelismConfig
    trial_metrics: TrialMetrics
    telemetry: dict[str, float | int | None] = field(default_factory=dict)
    nccl: dict[str, float | int | None] = field(default_factory=dict)
    adapter_metrics: dict[str, float] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "optimization_id": self.optimization_id,
            "profile_id": self.profile_id,
            "config": self.config.to_dict(),
            "trial_metrics": self.trial_metrics.to_dict(),
            "telemetry": dict(self.telemetry),
            "nccl": dict(self.nccl),
            "adapter_metrics": dict(self.adapter_metrics),
            "findings": list(self.findings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrelationReport:
        return cls(
            id=str(data["id"]),
            optimization_id=str(data["optimization_id"]),
            profile_id=str(data["profile_id"]),
            config=ParallelismConfig.from_dict(data["config"]),
            trial_metrics=TrialMetrics.from_dict(data["trial_metrics"]),
            telemetry={
                str(key): value
                for key, value in data.get("telemetry", {}).items()
                if value is None or isinstance(value, (int, float))
            },
            nccl={
                str(key): value
                for key, value in data.get("nccl", {}).items()
                if value is None or isinstance(value, (int, float))
            },
            adapter_metrics={
                str(key): float(value)
                for key, value in data.get("adapter_metrics", {}).items()
                if isinstance(value, (int, float))
            },
            findings=[str(item) for item in data.get("findings", [])],
        )
