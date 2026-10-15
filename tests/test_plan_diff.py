from adaptive_gpu_orchestrator.controller import DeploymentPlanner, SchedulerBackend
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig
from adaptive_gpu_orchestrator.validation import diff_plans


def test_plan_diff_records_configuration_and_world_size_changes():
    planner = DeploymentPlanner()
    baseline = planner.plan(
        config=ParallelismConfig(tensor_parallel=2, batch_size=2),
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=4,
    )
    candidate = planner.plan(
        config=ParallelismConfig(tensor_parallel=4, batch_size=4),
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=4,
    )

    diff = diff_plans(baseline, candidate)

    assert diff.changed is True
    assert diff.changes["world_size"] == {"before": 2, "after": 4}
    assert diff.changes["config"]["before"]["batch_size"] == 2
    assert diff.changes["config"]["after"]["batch_size"] == 4


def test_plan_diff_empty_for_equivalent_plans_except_ids_and_warnings():
    planner = DeploymentPlanner()
    baseline = planner.plan(
        config=ParallelismConfig(tensor_parallel=2),
        backend=SchedulerBackend.KUBERNETES,
        command=["python", "serve.py"],
        available_gpus=2,
    )
    candidate = planner.plan(
        config=ParallelismConfig(tensor_parallel=2),
        backend=SchedulerBackend.KUBERNETES,
        command=["python", "serve.py"],
        available_gpus=2,
    )

    assert diff_plans(baseline, candidate).changed is False
