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


def save_run(root, identifier, throughput, latency, energy, tp):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            energy_joules=energy,
        ),
        score=1.0,
    )
    OptimizationStore(root).save(
        OptimizationResult(
            id=identifier,
            objective=OptimizationObjective.BALANCED,
            constraints=OptimizationConstraints(),
            trials=[trial],
            best=trial,
        )
    )


def test_study_cli_compares_groups_and_persists_report(tmp_path, capsys):
    optimizations = tmp_path / "optimizations"
    studies = tmp_path / "studies"

    save_run(optimizations, "a1", 90, 105, 1100, 1)
    save_run(optimizations, "a2", 100, 100, 1000, 1)
    save_run(optimizations, "a3", 110, 95, 900, 2)
    save_run(optimizations, "b1", 110, 92, 960, 2)
    save_run(optimizations, "b2", 120, 90, 950, 2)
    save_run(optimizations, "b3", 130, 88, 940, 2)

    code = main(
        [
            "study",
            "analyze",
            "--group",
            "h100=a1,a2,a3",
            "--group",
            "h200=b1,b2,b3",
            "--optimization-store",
            str(optimizations),
            "--store-dir",
            str(studies),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Study:" in captured.out
    assert "Group h100" in captured.out
    assert "Group h200" in captured.out
    assert "h200 vs h100 throughput_per_s" in captured.out
    assert list(studies.glob("*.json"))

    list_code = main(
        [
            "studies",
            "list",
            "--store-dir",
            str(studies),
        ]
    )
    listed = capsys.readouterr()
    assert list_code == 0
    assert "groups=h100,h200" in listed.out
