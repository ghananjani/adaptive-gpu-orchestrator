from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from adaptive_gpu_orchestrator.validation import ValidationPolicy, ValidationReport


class AutopilotDecision(str, Enum):
    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class AutopilotPolicy:
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    min_study_replicates: int = 3
    min_config_stability: float = 0.67
    require_study: bool = False
    require_correlation: bool = False
    require_timeseries: bool = False
    require_pareto_candidate: bool = False
    hold_on_stragglers: bool = True
    max_memory_pressure_pct: float = 95.0
    max_temperature_c: float = 85.0
    require_plan_change: bool = True
    recommend_only: bool = True

    def __post_init__(self) -> None:
        if self.min_study_replicates <= 0:
            raise ValueError("min_study_replicates must be positive")
        if not 0 <= self.min_config_stability <= 1:
            raise ValueError("min_config_stability must be between 0 and 1")
        if self.max_memory_pressure_pct <= 0:
            raise ValueError("max_memory_pressure_pct must be positive")
        if self.max_temperature_c <= 0:
            raise ValueError("max_temperature_c must be positive")
        if not self.recommend_only:
            raise ValueError("v1.0 autopilot must remain recommend-only")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["validation"] = self.validation.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    target_config_label: str
    baseline_plan: dict[str, Any]
    trigger_reasons: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_config_label": self.target_config_label,
            "baseline_plan": dict(self.baseline_plan),
            "trigger_reasons": list(self.trigger_reasons),
            "steps": list(self.steps),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RollbackPlan:
        return cls(
            target_config_label=str(data["target_config_label"]),
            baseline_plan=dict(data["baseline_plan"]),
            trigger_reasons=tuple(
                str(item)
                for item in data.get("trigger_reasons", [])
            ),
            steps=tuple(str(item) for item in data.get("steps", [])),
        )


@dataclass(slots=True)
class AutopilotReport:
    baseline_optimization_id: str
    candidate_optimization_id: str
    decision: AutopilotDecision
    policy: AutopilotPolicy
    validation: ValidationReport
    baseline_plan: dict[str, Any]
    candidate_plan: dict[str, Any]
    plan_diff: dict[str, Any]
    rollback: RollbackPlan
    study_id: str | None = None
    correlation_id: str | None = None
    timeseries_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def promotable(self) -> bool:
        return self.decision is AutopilotDecision.PROMOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "baseline_optimization_id": self.baseline_optimization_id,
            "candidate_optimization_id": self.candidate_optimization_id,
            "decision": self.decision.value,
            "promotable": self.promotable,
            "policy": self.policy.to_dict(),
            "validation": self.validation.to_dict(),
            "baseline_plan": dict(self.baseline_plan),
            "candidate_plan": dict(self.candidate_plan),
            "plan_diff": dict(self.plan_diff),
            "rollback": self.rollback.to_dict(),
            "study_id": self.study_id,
            "correlation_id": self.correlation_id,
            "timeseries_id": self.timeseries_id,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutopilotReport:
        policy_data = dict(data.get("policy", {}))
        validation_data = dict(policy_data.pop("validation", {}))

        plan_diff = dict(data["plan_diff"])
        changes = dict(plan_diff.get("changes", {}))
        rank_change = changes.get("rank_to_gpu")
        if isinstance(rank_change, dict):
            normalized_rank_change = dict(rank_change)
            for side in ("before", "after"):
                mapping = normalized_rank_change.get(side)
                if isinstance(mapping, dict):
                    normalized_rank_change[side] = {
                        int(key) if str(key).isdigit() else key: value
                        for key, value in mapping.items()
                    }
            changes["rank_to_gpu"] = normalized_rank_change
        plan_diff["changes"] = changes

        return cls(
            id=str(data["id"]),
            baseline_optimization_id=str(data["baseline_optimization_id"]),
            candidate_optimization_id=str(data["candidate_optimization_id"]),
            decision=AutopilotDecision(data["decision"]),
            policy=AutopilotPolicy(
                validation=ValidationPolicy(**validation_data),
                **policy_data,
            ),
            validation=ValidationReport.from_dict(data["validation"]),
            baseline_plan=dict(data["baseline_plan"]),
            candidate_plan=dict(data["candidate_plan"]),
            plan_diff=plan_diff,
            rollback=RollbackPlan.from_dict(data["rollback"]),
            study_id=data.get("study_id"),
            correlation_id=data.get("correlation_id"),
            timeseries_id=data.get("timeseries_id"),
            reasons=[str(item) for item in data.get("reasons", [])],
            warnings=[str(item) for item in data.get("warnings", [])],
        )
