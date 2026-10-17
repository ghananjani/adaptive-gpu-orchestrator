import math

import pytest

from adaptive_gpu_orchestrator.modeling.evaluation import evaluate_hardware_transfer


def rows(platforms=("H100", "H200")):
    result = []
    for platform, offset in zip(platforms, (0.0, 25.0)):
        for tp in (1, 2, 4, 8):
            result.append({
                "gpu_models": platform,
                "exit_code": 0,
                "tensor_parallel": tp,
                "pipeline_parallel": 1,
                "data_parallel": 1,
                "batch_size": 1,
                "concurrency": 1,
                "throughput_per_s": 100.0 + 20.0 * math.log2(tp) + offset,
            })
    return result


def test_holdout_is_by_platform_and_deterministic():
    first = evaluate_hardware_transfer(rows(), ridge_lambda=1e-6)
    second = evaluate_hardware_transfer(reversed(rows()), ridge_lambda=1e-6)
    assert first.to_dict() == second.to_dict()
    assert first.sample_count == 8
    assert tuple(f.held_out_platform for f in first.folds) == ("H100", "H200")
    assert all(f.train_samples == 4 and f.test_samples == 4 for f in first.folds)
    assert first.pooled_rmse == pytest.approx(25.0, abs=0.05)
    assert first.pooled_mae == pytest.approx(25.0, abs=0.05)
    assert all(f.mean_baseline_rmse > f.rmse for f in first.folds)


def test_rejects_single_platform():
    with pytest.raises(ValueError, match="at least two platforms"):
        evaluate_hardware_transfer(rows(("H100",)))


def test_rejects_untrainable_holdout():
    samples = rows()
    samples = [r for r in samples if r["gpu_models"] == "H100"] + [
        next(r for r in samples if r["gpu_models"] == "H200")
    ]
    with pytest.raises(ValueError, match="insufficient training diversity"):
        evaluate_hardware_transfer(samples)


def test_nonfinite_targets_and_failed_trials_excluded():
    samples = rows()
    samples[0]["throughput_per_s"] = float("nan")
    samples[1]["exit_code"] = 1
    outcome = evaluate_hardware_transfer(samples)
    assert outcome.sample_count == 6


def test_validates_ridge_and_sample_count():
    with pytest.raises(ValueError, match="finite"):
        evaluate_hardware_transfer(rows(), ridge_lambda=float("nan"))
    with pytest.raises(ValueError, match="at least two"):
        evaluate_hardware_transfer(rows(), min_train_samples=1)
