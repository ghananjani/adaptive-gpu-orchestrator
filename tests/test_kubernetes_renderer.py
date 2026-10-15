from adaptive_gpu_orchestrator.controller import DeploymentPlanner, SchedulerBackend
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig
from adaptive_gpu_orchestrator.schedulers import KubernetesRenderer


def test_kubernetes_job_requests_plan_world_size():
    plan = DeploymentPlanner().plan(
        config=ParallelismConfig(tensor_parallel=4, batch_size=8),
        backend=SchedulerBackend.KUBERNETES,
        command=["python", "serve.py"],
        available_gpus=4,
        node_selector={"accelerator": "h100"},
    )

    manifest = KubernetesRenderer(
        image="example/llm:latest",
        job_name="llm-canary",
    ).render(plan)

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert manifest["kind"] == "Job"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 4
    assert manifest["spec"]["template"]["spec"]["nodeSelector"] == {
        "accelerator": "h100"
    }
    assert any(item["name"] == "AGO_WORLD_SIZE" for item in container["env"])
