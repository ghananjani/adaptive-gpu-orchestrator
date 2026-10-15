from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import uuid4

from adaptive_gpu_orchestrator.models import TopologyLink
from adaptive_gpu_orchestrator.optimizer import (
    CommunicationEdge,
    ParallelismConfig,
    optimize_placement,
)

from .models import DeploymentPlan, DeploymentPolicy, SchedulerBackend


@dataclass(slots=True)
class DeploymentPlanner:
    policy: DeploymentPolicy = field(default_factory=DeploymentPolicy)

    def _validate(
        self,
        config: ParallelismConfig,
        backend: SchedulerBackend,
        command: list[str],
        available_gpus: int,
    ) -> None:
        if not command:
            raise ValueError("deployment command cannot be empty")
        if available_gpus <= 0:
            raise ValueError("available_gpus must be positive")
        if config.world_size > available_gpus:
            raise ValueError(
                f"configuration needs {config.world_size} GPUs but only "
                f"{available_gpus} are available"
            )
        if self.policy.max_gpus is not None and config.world_size > self.policy.max_gpus:
            raise ValueError(
                f"configuration needs {config.world_size} GPUs but policy allows "
                f"at most {self.policy.max_gpus}"
            )
        if self.policy.require_exact_gpu_count and config.world_size != available_gpus:
            raise ValueError(
                "policy requires the configuration world size to match available_gpus"
            )
        if backend not in SchedulerBackend:
            raise ValueError(f"unsupported scheduler backend: {backend}")

    @staticmethod
    def _environment(
        config: ParallelismConfig,
        rank_to_gpu: dict[int, int],
    ) -> dict[str, str]:
        return {
            "AGO_TP": str(config.tensor_parallel),
            "AGO_PP": str(config.pipeline_parallel),
            "AGO_DP": str(config.data_parallel),
            "AGO_BATCH_SIZE": str(config.batch_size),
            "AGO_CONCURRENCY": str(config.concurrency),
            "AGO_WORLD_SIZE": str(config.world_size),
            "AGO_CONFIG_LABEL": config.label(),
            "AGO_RANK_TO_GPU": json.dumps(rank_to_gpu, sort_keys=True),
        }

    def plan(
        self,
        *,
        config: ParallelismConfig,
        backend: SchedulerBackend,
        command: list[str],
        available_gpus: int,
        gpu_indices: list[int] | None = None,
        communication: list[CommunicationEdge] | None = None,
        topology: list[TopologyLink] | None = None,
        node_selector: dict[str, str] | None = None,
    ) -> DeploymentPlan:
        self._validate(config, backend, command, available_gpus)

        physical = gpu_indices or list(range(available_gpus))
        if len(set(physical)) != len(physical):
            raise ValueError("gpu_indices must be unique")
        if len(physical) < config.world_size:
            raise ValueError("gpu_indices does not contain enough devices")

        ranks = list(range(config.world_size))
        selected_gpus = physical[: config.world_size]

        warnings: list[str] = []
        if communication and topology:
            placement = optimize_placement(
                ranks,
                physical,
                communication,
                topology,
            )
            rank_to_gpu = placement.rank_to_gpu
        else:
            rank_to_gpu = dict(zip(ranks, selected_gpus, strict=True))
            warnings.append(
                "No communication graph/topology pair supplied; using sequential GPU placement."
            )

        if self.policy.recommend_only:
            warnings.append(
                "Recommend-only policy is active; this plan does not mutate scheduler state."
            )

        return DeploymentPlan(
            id=uuid4().hex[:12],
            backend=backend,
            config=config,
            command=list(command),
            available_gpus=available_gpus,
            rank_to_gpu=rank_to_gpu,
            env=self._environment(config, rank_to_gpu),
            node_selector=dict(node_selector or {}),
            warnings=warnings,
            recommend_only=self.policy.recommend_only,
        )
