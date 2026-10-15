from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from adaptive_gpu_orchestrator.optimizer import ParallelismConfig


class SchedulerBackend(str, Enum):
    KUBERNETES = "kubernetes"
    SLURM = "slurm"


@dataclass(frozen=True, slots=True)
class DeploymentPolicy:
    max_gpus: int | None = None
    require_exact_gpu_count: bool = False
    recommend_only: bool = True

    def __post_init__(self) -> None:
        if self.max_gpus is not None and self.max_gpus <= 0:
            raise ValueError("max_gpus must be positive when configured")


@dataclass(slots=True)
class DeploymentPlan:
    id: str
    backend: SchedulerBackend
    config: ParallelismConfig
    command: list[str]
    available_gpus: int
    rank_to_gpu: dict[int, int]
    env: dict[str, str] = field(default_factory=dict)
    node_selector: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    recommend_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend.value,
            "config": self.config.to_dict(),
            "world_size": self.config.world_size,
            "command": list(self.command),
            "available_gpus": self.available_gpus,
            "rank_to_gpu": {str(k): v for k, v in self.rank_to_gpu.items()},
            "env": dict(self.env),
            "node_selector": dict(self.node_selector),
            "warnings": list(self.warnings),
            "recommend_only": self.recommend_only,
        }
