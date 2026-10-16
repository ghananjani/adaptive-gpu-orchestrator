from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.controller import DeploymentPlan, SchedulerBackend


@dataclass(slots=True)
class KubernetesJobSetRenderer:
    image: str
    job_name: str = "ago-workload"
    nodes: int = 1
    gpus_per_node: int = 1
    queue_name: str | None = None
    master_port: int = 29500
    replicated_job_name: str = "workers"
    service_account_name: str | None = None

    def _validate(self, plan: DeploymentPlan) -> None:
        if not self.image:
            raise ValueError("container image is required")
        if plan.backend is not SchedulerBackend.KUBERNETES:
            raise ValueError("JobSet renderer requires a Kubernetes deployment plan")
        if self.nodes <= 0:
            raise ValueError("nodes must be positive")
        if self.gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        if self.master_port <= 0 or self.master_port > 65535:
            raise ValueError("master_port must be between 1 and 65535")
        if not self.replicated_job_name:
            raise ValueError("replicated_job_name cannot be empty")

        requested = self.nodes * self.gpus_per_node
        if requested != plan.config.world_size:
            raise ValueError(
                f"JobSet shape requests {requested} GPUs "
                f"({self.nodes} nodes x {self.gpus_per_node}) but "
                f"configuration world size is {plan.config.world_size}"
            )
        if requested > plan.available_gpus:
            raise ValueError(
                f"JobSet requests {requested} GPUs but only "
                f"{plan.available_gpus} are available"
            )

    def render(self, plan: DeploymentPlan) -> dict[str, Any]:
        self._validate(plan)

        labels = {"ago.plan-id": plan.id}
        if self.queue_name:
            labels["kueue.x-k8s.io/queue-name"] = self.queue_name

        master_addr = (
            f"{self.job_name}-{self.replicated_job_name}-0-0."
            f"{self.job_name}"
        )
        env: list[dict[str, Any]] = [
            {"name": key, "value": value}
            for key, value in sorted(plan.env.items())
        ]
        env.extend(
            [
                {"name": "AGO_NODE_COUNT", "value": str(self.nodes)},
                {"name": "AGO_GPUS_PER_NODE", "value": str(self.gpus_per_node)},
                {"name": "AGO_MASTER_ADDR", "value": master_addr},
                {"name": "MASTER_ADDR", "value": master_addr},
                {"name": "MASTER_PORT", "value": str(self.master_port)},
                {
                    "name": "AGO_NODE_RANK",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": (
                                "metadata.labels['jobset.sigs.k8s.io/job-index']"
                            )
                        }
                    },
                },
            ]
        )

        container = {
            "name": "workload",
            "image": self.image,
            "command": plan.command,
            "env": env,
            "resources": {
                "requests": {"nvidia.com/gpu": self.gpus_per_node},
                "limits": {"nvidia.com/gpu": self.gpus_per_node},
            },
        }
        pod_spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [container],
        }
        if plan.node_selector:
            pod_spec["nodeSelector"] = dict(plan.node_selector)
        if self.service_account_name:
            pod_spec["serviceAccountName"] = self.service_account_name

        return {
            "apiVersion": "jobset.x-k8s.io/v1alpha2",
            "kind": "JobSet",
            "metadata": {
                "name": self.job_name,
                "labels": labels,
            },
            "spec": {
                "replicatedJobs": [
                    {
                        "name": self.replicated_job_name,
                        "replicas": self.nodes,
                        "template": {
                            "spec": {
                                "parallelism": 1,
                                "completions": 1,
                                "backoffLimit": 0,
                                "template": {
                                    "metadata": {
                                        "labels": {"ago.plan-id": plan.id}
                                    },
                                    "spec": pod_spec,
                                },
                            }
                        },
                    }
                ]
            },
        }
