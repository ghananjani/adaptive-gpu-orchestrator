from adaptive_gpu_orchestrator.cli import main
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStore,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)


def save_run(root, identifier, tp, throughput):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=100.0,
            energy_joules=1000.0,
            memory_pressure_pct=50.0,
        ),
        score=1.0,
    )
    OptimizationStore(root).save(
        OptimizationResult(
            id=identifier,
            objective=OptimizationObjective.THROUGHPUT,
            constraints=OptimizationConstraints(),
            trials=[trial],
            best=trial,
        )
    )


def test_autopilot_cli_promotes_and_persists_decision(tmp_path, capsys):
    optimizations = tmp_path / "optimizations"
    reports = tmp_path / "autopilot"

    save_run(optimizations, "baseline", 1, 100.0)
    save_run(optimizations, "candidate", 2, 120.0)

    code = main(
        [
            "autopilot",
            "decide",
            "--baseline-optimization",
            "baseline",
            "--candidate-optimization",
            "candidate",
            "--backend",
            "slurm",
            "--available-gpus",
            "2",
            "--optimization-store",
            str(optimizations),
            "--store-dir",
            str(reports),
            "--min-throughput-gain",
            "5",
            "--",
            "python",
            "train.py",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "decision=promote" in captured.out
    assert "rollback_target=tp1-pp1-dp1-b1-c1" in captured.out
    assert list(reports.glob("*.json"))

    list_code = main(
        [
            "autopilot",
            "list",
            "--store-dir",
            str(reports),
        ]
    )
    listed = capsys.readouterr()
    assert list_code == 0
    assert "baseline -> candidate" in listed.out


def test_autopilot_cli_returns_hold_code_when_required_evidence_missing(
    tmp_path,
    capsys,
):
    optimizations = tmp_path / "optimizations"
    reports = tmp_path / "autopilot"

    save_run(optimizations, "baseline", 1, 100.0)
    save_run(optimizations, "candidate", 2, 120.0)

    code = main(
        [
            "autopilot",
            "decide",
            "--baseline-optimization",
            "baseline",
            "--candidate-optimization",
            "candidate",
            "--backend",
            "slurm",
            "--available-gpus",
            "2",
            "--optimization-store",
            str(optimizations),
            "--store-dir",
            str(reports),
            "--require-study",
            "--",
            "python",
            "train.py",
        ]
    )

    captured = capsys.readouterr()
    assert code == 4
    assert "decision=hold" in captured.out
    assert "required statistical study evidence is missing" in captured.out
