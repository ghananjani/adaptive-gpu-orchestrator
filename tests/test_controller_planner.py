import pytest

from adaptive_gpu_orchestrator.controller import (
    DeploymentPlanner,
    DeploymentPolicy,
    SchedulerBackend,
)
from adaptive_gpu_orchestrator.models import TopologyLink
from adaptive_gpu_orchestrator.optimizer import CommunicationEdge, ParallelismConfig


def test_policy_rejects_gpu_budget_violation():
    planner = DeploymentPlanner(DeploymentPolicy(max_gpus=2))
    config = ParallelismConfig(tensor_parallel=4)

    with pytest.raises(ValueError):
        planner.plan(
            config=config,
            backend=SchedulerBackend.SLURM,
            command=["python", "train.py"],
            available_gpus=4,
        )


def test_planner_uses_topology_aware_placement():
    planner = DeploymentPlanner()
    config = ParallelismConfig(tensor_parallel=2)
    topology = [
        TopologyLink(0, 1, "NV4", 40, "NV4"),
        TopologyLink(0, 2, "SYS", 1, "SYS"),
        TopologyLink(1, 2, "SYS", 1, "SYS"),
    ]
    communication = [CommunicationEdge(0, 1, 100.0)]

    plan = planner.plan(
        config=config,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=3,
        gpu_indices=[0, 1, 2],
        topology=topology,
        communication=communication,
    )

    assert set(plan.rank_to_gpu.values()) == {0, 1}
    assert plan.recommend_only is True
    assert "AGO_RANK_TO_GPU" in plan.env


def test_exact_gpu_policy():
    planner = DeploymentPlanner(
        DeploymentPolicy(require_exact_gpu_count=True)
    )

    with pytest.raises(ValueError):
        planner.plan(
            config=ParallelismConfig(tensor_parallel=2),
            backend=SchedulerBackend.KUBERNETES,
            command=["python", "serve.py"],
            available_gpus=4,
        )
