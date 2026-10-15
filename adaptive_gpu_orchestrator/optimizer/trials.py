from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig


class OptimizationObjective(str, Enum):
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ENERGY = "energy"
    BALANCED = "balanced"


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    throughput_per_s: float | None = None
    p99_latency_ms: float | None = None
    energy_joules: float | None = None
    gpu_util_pct: float | None = None
    memory_pressure_pct: float | None = None
    duration_s: float | None = None
    exit_code: int = 0
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialMetrics:
        def optional_float(key: str) -> float | None:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            throughput_per_s=optional_float("throughput_per_s"),
            p99_latency_ms=optional_float("p99_latency_ms"),
            energy_joules=optional_float("energy_joules"),
            gpu_util_pct=optional_float("gpu_util_pct"),
            memory_pressure_pct=optional_float("memory_pressure_pct"),
            duration_s=optional_float("duration_s"),
            exit_code=int(data.get("exit_code", 0)),
            extras={
                str(key): float(value)
                for key, value in data.get("extras", {}).items()
                if isinstance(value, (int, float))
            },
        )


@dataclass(frozen=True, slots=True)
class OptimizationConstraints:
    max_p99_latency_ms: float | None = None
    max_energy_joules: float | None = None
    min_throughput_per_s: float | None = None
    max_memory_pressure_pct: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(slots=True)
class TrialResult:
    config: ParallelismConfig
    metrics: TrialMetrics
    feasible: bool = True
    score: float | None = None
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "feasible": self.feasible,
            "score": self.score,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(slots=True)
class OptimizationResult:
    objective: OptimizationObjective
    constraints: OptimizationConstraints
    trials: list[TrialResult]
    best: TrialResult | None = None
    baseline: TrialResult | None = None
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective.value,
            "constraints": self.constraints.to_dict(),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "best": self.best.to_dict() if self.best else None,
            "trials": [trial.to_dict() for trial in self.trials],
        }
