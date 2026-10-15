from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adaptive_gpu_orchestrator.controller import DeploymentPlan


@dataclass(slots=True)
class PlanDiff:
    baseline_plan_id: str
    candidate_plan_id: str
    changes: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_plan_id": self.baseline_plan_id,
            "candidate_plan_id": self.candidate_plan_id,
            "changed": self.changed,
            "changes": self.changes,
        }


def diff_plans(
    baseline: DeploymentPlan,
    candidate: DeploymentPlan,
) -> PlanDiff:
    changes: dict[str, Any] = {}

    def add(name: str, before: Any, after: Any) -> None:
        if before != after:
            changes[name] = {"before": before, "after": after}

    add("backend", baseline.backend.value, candidate.backend.value)
    add("config", baseline.config.to_dict(), candidate.config.to_dict())
    add("world_size", baseline.config.world_size, candidate.config.world_size)
    add("command", baseline.command, candidate.command)
    add("available_gpus", baseline.available_gpus, candidate.available_gpus)
    add("rank_to_gpu", baseline.rank_to_gpu, candidate.rank_to_gpu)
    add("node_selector", baseline.node_selector, candidate.node_selector)
    add("env", baseline.env, candidate.env)

    return PlanDiff(
        baseline_plan_id=baseline.id,
        candidate_plan_id=candidate.id,
        changes=changes,
    )
