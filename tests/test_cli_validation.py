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


def save_optimization(root, identifier, *, tp, throughput, latency):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=tp),
        metrics=TrialMetrics(
            throughput_per_s=throughput,
            p99_latency_ms=latency,
            exit_code=0,
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


def test_validate_accepts_and_persists_report(tmp_path, capsys):
    optimizations = tmp_path / "optimizations"
    validations = tmp_path / "validations"
    save_optimization(
        optimizations,
        "baseline",
        tp=1,
        throughput=1000,
        latency=100,
    )
    save_optimization(
        optimizations,
        "candidate",
        tp=2,
        throughput=1200,
        latency=103,
    )

    code = main(
        [
            "validate",
            "--baseline-optimization",
            "baseline",
            "--candidate-optimization",
            "candidate",
            "--optimization-store",
            str(optimizations),
            "--validation-store",
            str(validations),
            "--min-throughput-gain",
            "10",
            "--max-p99-regression",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "decision=accept" in captured.out
    assert list(validations.glob("*.json"))


def test_validate_rejects_bad_candidate(tmp_path, capsys):
    optimizations = tmp_path / "optimizations"
    validations = tmp_path / "validations"
    save_optimization(
        optimizations,
        "baseline",
        tp=1,
        throughput=1000,
        latency=100,
    )
    save_optimization(
        optimizations,
        "candidate",
        tp=2,
        throughput=1050,
        latency=120,
    )

    code = main(
        [
            "validate",
            "--baseline-optimization",
            "baseline",
            "--candidate-optimization",
            "candidate",
            "--optimization-store",
            str(optimizations),
            "--validation-store",
            str(validations),
            "--min-throughput-gain",
            "10",
            "--max-p99-regression",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert code == 3
    assert "decision=reject" in captured.out
    assert "rollback_recommended=True" in captured.out
