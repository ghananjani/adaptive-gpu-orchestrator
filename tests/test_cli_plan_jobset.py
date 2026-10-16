import json

from adaptive_gpu_orchestrator.cli import main


def test_plan_kubernetes_jobset_json(capsys):
    code = main(
        [
            "--json",
            "plan",
            "--backend",
            "kubernetes",
            "--available-gpus",
            "8",
            "--tp",
            "8",
            "--image",
            "example/train:latest",
            "--job-name",
            "llm-train",
            "--kubernetes-mode",
            "jobset",
            "--nodes",
            "2",
            "--gpus-per-node",
            "4",
            "--queue-name",
            "gpu-queue",
            "--master-port",
            "29501",
            "--node-selector",
            "accelerator=h100",
            "--",
            "python",
            "train.py",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    artifact = payload["artifact"]

    assert artifact["apiVersion"] == "jobset.x-k8s.io/v1alpha2"
    assert artifact["kind"] == "JobSet"
    assert (
        artifact["metadata"]["labels"]["kueue.x-k8s.io/queue-name"]
        == "gpu-queue"
    )

    replicated = artifact["spec"]["replicatedJobs"][0]
    assert replicated["replicas"] == 2
    container = (
        replicated["template"]["spec"]["template"]["spec"]["containers"][0]
    )
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 4

    env = {item["name"]: item for item in container["env"]}
    assert env["AGO_NODE_COUNT"]["value"] == "2"
    assert env["AGO_GPUS_PER_NODE"]["value"] == "4"
    assert env["MASTER_PORT"]["value"] == "29501"


def test_single_job_rejects_multi_node_shape(capsys):
    code = main(
        [
            "plan",
            "--backend",
            "kubernetes",
            "--available-gpus",
            "8",
            "--tp",
            "8",
            "--image",
            "example/train:latest",
            "--nodes",
            "2",
            "--",
            "python",
            "train.py",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "use --kubernetes-mode jobset" in captured.err
