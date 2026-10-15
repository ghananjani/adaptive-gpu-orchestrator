import json

from adaptive_gpu_orchestrator.cli import main


def test_plan_slurm_renders_recommend_only_script(capsys):
    code = main(
        [
            "plan",
            "--backend",
            "slurm",
            "--available-gpus",
            "4",
            "--tp",
            "4",
            "--nodes",
            "1",
            "--gpus-per-node",
            "4",
            "--",
            "python",
            "train.py",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "#SBATCH --gpus-per-node=4" in captured.out
    assert "srun python train.py" in captured.out
    assert "Recommend-only policy is active" in captured.err


def test_plan_kubernetes_json(capsys):
    code = main(
        [
            "--json",
            "plan",
            "--backend",
            "kubernetes",
            "--available-gpus",
            "2",
            "--tp",
            "2",
            "--image",
            "example/llm:latest",
            "--node-selector",
            "accelerator=h100",
            "--",
            "python",
            "serve.py",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["plan"]["recommend_only"] is True
    assert payload["artifact"]["kind"] == "Job"
    assert (
        payload["artifact"]["spec"]["template"]["spec"]["nodeSelector"]["accelerator"]
        == "h100"
    )
