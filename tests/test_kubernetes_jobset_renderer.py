import pytest

from adaptive_gpu_orchestrator.controller import DeploymentPlanner, SchedulerBackend
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig
from adaptive_gpu_orchestrator.schedulers import KubernetesJobSetRenderer


def plan(world_size=8):
    return DeploymentPlanner().plan(
        config=ParallelismConfig(tensor_parallel=world_size),
        backend=SchedulerBackend.KUBERNETES,
        command=["python", "train.py"],
        available_gpus=world_size,
        node_selector={"accelerator": "h100"},
    )


def test_jobset_renders_multinode_gpu_shape_and_kueue_metadata():
    manifest = KubernetesJobSetRenderer(
        image="example/train:latest",
        job_name="llm-train",
        nodes=2,
        gpus_per_node=4,
        queue_name="gpu-queue",
        master_port=29501,
    ).render(plan())

    assert manifest["apiVersion"] == "jobset.x-k8s.io/v1alpha2"
    assert manifest["kind"] == "JobSet"
    assert (
        manifest["metadata"]["labels"]["kueue.x-k8s.io/queue-name"]
        == "gpu-queue"
    )

    replicated = manifest["spec"]["replicatedJobs"][0]
    assert replicated["name"] == "workers"
    assert replicated["replicas"] == 2
    assert replicated["template"]["spec"]["parallelism"] == 1
    assert replicated["template"]["spec"]["completions"] == 1

    pod_spec = replicated["template"]["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"] == {"accelerator": "h100"}
    container = pod_spec["containers"][0]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 4
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 4

    env = {item["name"]: item for item in container["env"]}
    assert env["AGO_NODE_COUNT"]["value"] == "2"
    assert env["AGO_GPUS_PER_NODE"]["value"] == "4"
    assert env["MASTER_PORT"]["value"] == "29501"
    assert env["MASTER_ADDR"]["value"] == "llm-train-workers-0-0.llm-train"
    assert env["AGO_NODE_RANK"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.labels['jobset.sigs.k8s.io/job-index']"
    )


def test_jobset_supports_service_account():
    manifest = KubernetesJobSetRenderer(
        image="example/train:latest",
        nodes=2,
        gpus_per_node=4,
        service_account_name="trainer",
    ).render(plan())

    pod_spec = (
        manifest["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    )
    assert pod_spec["serviceAccountName"] == "trainer"


def test_jobset_rejects_shape_that_does_not_match_world_size():
    with pytest.raises(ValueError, match="configuration world size is 8"):
        KubernetesJobSetRenderer(
            image="example/train:latest",
            nodes=3,
            gpus_per_node=2,
        ).render(plan())


def test_jobset_rejects_non_kubernetes_plan():
    slurm_plan = DeploymentPlanner().plan(
        config=ParallelismConfig(tensor_parallel=2),
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=2,
    )

    with pytest.raises(ValueError, match="requires a Kubernetes deployment plan"):
        KubernetesJobSetRenderer(
            image="example/train:latest",
            nodes=1,
            gpus_per_node=2,
        ).render(slurm_plan)
