from __future__ import annotations

import math
import shlex
from dataclasses import dataclass

from adaptive_gpu_orchestrator.controller import DeploymentPlan


@dataclass(slots=True)
class SlurmRenderer:
    job_name: str = "ago-workload"
    nodes: int = 1
    gpus_per_node: int | None = None
    partition: str | None = None

    def render(self, plan: DeploymentPlan) -> str:
        if self.nodes <= 0:
            raise ValueError("nodes must be positive")

        gpus_per_node = self.gpus_per_node
        if gpus_per_node is None:
            gpus_per_node = math.ceil(plan.config.world_size / self.nodes)
        if gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        if self.nodes * gpus_per_node < plan.config.world_size:
            raise ValueError("Slurm allocation cannot satisfy the plan world size")

        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --gpus-per-node={gpus_per_node}",
            f"#SBATCH --ntasks={plan.config.world_size}",
        ]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")

        lines.extend(["", "set -euo pipefail", ""])
        for key, value in sorted(plan.env.items()):
            lines.append(f"export {key}={shlex.quote(value)}")

        lines.append("")
        command = " ".join(shlex.quote(part) for part in plan.command)
        lines.append(f"srun {command}")
        lines.append("")
        return "\n".join(lines)
