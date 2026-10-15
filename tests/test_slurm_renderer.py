from adaptive_gpu_orchestrator.controller import DeploymentPlanner, SchedulerBackend
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig
from adaptive_gpu_orchestrator.schedulers import SlurmRenderer


def test_slurm_directives_precede_shell_commands():
    plan = DeploymentPlanner().plan(
        config=ParallelismConfig(tensor_parallel=4),
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py", "--steps", "10"],
        available_gpus=4,
    )

    script = SlurmRenderer(
        job_name="train-canary",
        nodes=1,
        gpus_per_node=4,
        partition="gpu",
    ).render(plan)

    lines = script.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert lines[1].startswith("#SBATCH --job-name=")
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --partition=gpu" in script
    assert script.index("#SBATCH --ntasks=4") < script.index("set -euo pipefail")
    assert "srun python train.py --steps 10" in script
