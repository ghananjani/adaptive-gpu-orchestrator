import json

from adaptive_gpu_orchestrator.autopilot import (
    AutopilotPolicy,
    AutopilotStore,
    decide_autopilot,
)
from adaptive_gpu_orchestrator.cli import main
from adaptive_gpu_orchestrator.controller import SchedulerBackend
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStore,
    ParallelismConfig,
    TrialMetrics,
    TrialResult,
)
from adaptive_gpu_orchestrator.validation import ValidationPolicy


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
    result = OptimizationResult(
        id=identifier,
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        trials=[trial],
        best=trial,
    )
    OptimizationStore(root).save(result)
    return result


def test_bundle_cli_create_verify_list_and_detect_tampering(tmp_path, capsys):
    optimizations = tmp_path / "optimizations"
    autopilot_root = tmp_path / "autopilot"
    bundles = tmp_path / "bundles"

    baseline = save_run(optimizations, "baseline", 1, 100.0)
    candidate = save_run(optimizations, "candidate", 2, 120.0)
    report = decide_autopilot(
        baseline=baseline,
        candidate=candidate,
        backend=SchedulerBackend.SLURM,
        command=["python", "train.py"],
        available_gpus=2,
        policy=AutopilotPolicy(
            validation=ValidationPolicy(min_throughput_gain_pct=5)
        ),
    )
    AutopilotStore(autopilot_root).save(report)

    create_code = main(
        [
            "bundle",
            "create",
            "--autopilot",
            report.id,
            "--autopilot-store",
            str(autopilot_root),
            "--optimization-store",
            str(optimizations),
            "--store-dir",
            str(bundles),
            "--tag",
            "hardware=h100",
        ]
    )

    created = capsys.readouterr()
    assert create_code == 0
    assert "Integrity: valid=True" in created.out
    assert "artifacts=3" in created.out

    paths = list(bundles.glob("*.json"))
    assert len(paths) == 1
    bundle_id = paths[0].stem

    verify_code = main(
        [
            "bundle",
            "verify",
            bundle_id,
            "--store-dir",
            str(bundles),
        ]
    )
    verified = capsys.readouterr()
    assert verify_code == 0
    assert "Integrity: valid=True" in verified.out

    list_code = main(
        [
            "bundle",
            "list",
            "--store-dir",
            str(bundles),
        ]
    )
    listed = capsys.readouterr()
    assert list_code == 0
    assert f"autopilot={report.id}" in listed.out

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["artifacts"][1]["payload"]["id"] = "tampered"
    paths[0].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tampered_code = main(
        [
            "bundle",
            "verify",
            bundle_id,
            "--store-dir",
            str(bundles),
        ]
    )
    tampered = capsys.readouterr()
    assert tampered_code == 5
    assert "Integrity: valid=False" in tampered.out
    assert "optimization.baseline:baseline" in tampered.out
