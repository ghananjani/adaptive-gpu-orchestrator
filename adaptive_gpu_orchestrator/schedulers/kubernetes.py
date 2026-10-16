from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.controller import DeploymentPlan


@dataclass(slots=True)
class KubernetesRenderer:
    image: str
    job_name: str = "ago-workload"
    queue_name: str | None = None

    def render(self, plan: DeploymentPlan) -> dict[str, Any]:
        if not self.image:
            raise ValueError("container image is required")

        env = [
            {"name": key, "value": value}
            for key, value in sorted(plan.env.items())
        ]
        container = {
            "name": "workload",
            "image": self.image,
            "command": plan.command,
            "env": env,
            "resources": {
                "requests": {"nvidia.com/gpu": plan.config.world_size},
                "limits": {"nvidia.com/gpu": plan.config.world_size},
            },
        }
        pod_spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [container],
        }
        if plan.node_selector:
            pod_spec["nodeSelector"] = dict(plan.node_selector)

        labels = {"ago.plan-id": plan.id}
        if self.queue_name:
            labels["kueue.x-k8s.io/queue-name"] = self.queue_name

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.job_name,
                "labels": labels,
            },
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": {"ago.plan-id": plan.id}},
                    "spec": pod_spec,
                },
            },
        }
