from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig


class ValidationDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    min_throughput_gain_pct: float | None = None
    max_p99_latency_regression_pct: float | None = None
    max_energy_regression_pct: float | None = None
    max_memory_pressure_regression_pct: float | None = None
    rollback_on_reject: bool = True

    def __post_init__(self) -> None:
        values = (
            self.min_throughput_gain_pct,
            self.max_p99_latency_regression_pct,
            self.max_energy_regression_pct,
            self.max_memory_pressure_regression_pct,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("validation thresholds must be non-negative")

    def to_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricDelta:
    baseline: float
    candidate: float
    percent_change: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricDelta:
        return cls(
            baseline=float(data["baseline"]),
            candidate=float(data["candidate"]),
            percent_change=float(data["percent_change"]),
        )


@dataclass(slots=True)
class ValidationReport:
    id: str
    decision: ValidationDecision
    policy: ValidationPolicy
    baseline_config: ParallelismConfig
    candidate_config: ParallelismConfig
    deltas: dict[str, MetricDelta] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    rollback_recommended: bool = False

    @property
    def accepted(self) -> bool:
        return self.decision is ValidationDecision.ACCEPT

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "decision": self.decision.value,
            "accepted": self.accepted,
            "rollback_recommended": self.rollback_recommended,
            "policy": self.policy.to_dict(),
            "baseline_config": self.baseline_config.to_dict(),
            "candidate_config": self.candidate_config.to_dict(),
            "deltas": {
                key: delta.to_dict()
                for key, delta in self.deltas.items()
            },
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        policy_data = dict(data.get("policy", {}))
        return cls(
            id=str(data["id"]),
            decision=ValidationDecision(data["decision"]),
            policy=ValidationPolicy(**policy_data),
            baseline_config=ParallelismConfig.from_dict(data["baseline_config"]),
            candidate_config=ParallelismConfig.from_dict(data["candidate_config"]),
            deltas={
                key: MetricDelta.from_dict(value)
                for key, value in data.get("deltas", {}).items()
            },
            reasons=[str(item) for item in data.get("reasons", [])],
            rollback_recommended=bool(data.get("rollback_recommended", False)),
        )
