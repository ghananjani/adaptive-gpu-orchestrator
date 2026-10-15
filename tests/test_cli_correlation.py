from pathlib import Path

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
from adaptive_gpu_orchestrator.profiling import ExperimentStore, WorkloadProfile
from adaptive_gpu_orchestrator.telemetry import TelemetrySample


def save_inputs(optimization_root: Path, experiment_root: Path):
    trial = TrialResult(
        config=ParallelismConfig(tensor_parallel=2),
        metrics=TrialMetrics(
            throughput_per_s=100.0,
            extras={"communication_fraction_pct": 30.0},
        ),
        score=1.0,
    )
    OptimizationStore(optimization_root).save(
        OptimizationResult(
            id="opt1",
            objective=OptimizationObjective.THROUGHPUT,
            constraints=OptimizationConstraints(),
            trials=[trial],
            best=trial,
        )
    )
    ExperimentStore(experiment_root).save(
        WorkloadProfile(
            id="profile1",
            name="training",
            duration_s=2.0,
            samples=[
                TelemetrySample(
                    offset_s=0.0,
                    gpu_count=2,
                    gpu_util_mean_pct=55.0,
                    gpu_util_std_pct=10.0,
                    memory_pressure_mean_pct=70.0,
                    temperature_max_c=72.0,
                    power_ratio_max_pct=80.0,
                    power_draw_total_w=300.0,
                )
            ],
        )
    )


def test_correlate_cli_persists_report_with_nccl(tmp_path, capsys):
    optimizations = tmp_path / "optimizations"
    experiments = tmp_path / "experiments"
    correlations = tmp_path / "correlations"
    save_inputs(optimizations, experiments)

    nccl = tmp_path / "nccl.txt"
    nccl.write_text(
        "8388608 2097152 float sum -1 100.0 80.0 70.0\n",
        encoding="utf-8",
    )

    code = main(
        [
            "correlate",
            "--optimization",
            "opt1",
            "--profile",
            "profile1",
            "--optimization-store",
            str(optimizations),
            "--experiment-store",
            str(experiments),
            "--nccl-output",
            str(nccl),
            "--store-dir",
            str(correlations),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Correlation:" in captured.out
    assert "NCCL mean bus bandwidth baseline" in captured.out
    assert list(correlations.glob("*.json"))

    list_code = main(
        [
            "correlations",
            "--store-dir",
            str(correlations),
            "list",
        ]
    )
    list_output = capsys.readouterr()
    assert list_code == 0
    assert "optimization=opt1" in list_output.out
