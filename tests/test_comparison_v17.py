import math

import pytest

from adaptive_gpu_orchestrator.modeling.comparison import (
    LabeledHardwareTrial,
    compare_hardware_heldout_models,
)
from adaptive_gpu_orchestrator.modeling.hardware import HardwareFeatures
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig


def trials():
    values = []
    for index, name in enumerate(("H100-test", "H200-test", "H300-synthetic")):
        hardware = HardwareFeatures(
            gpu_count=8,
            memory_gib_per_gpu=64 + 16 * index,
            memory_bandwidth_gbps_per_gpu=1500 + index * 400,
            compute_tflops_per_gpu=40 + 10 * index,
            interconnect_bandwidth_gbps=300 + 100 * index,
        )
        for tp in (1, 2, 4, 8):
            config = ParallelismConfig(tensor_parallel=tp)
            target = 90 + 18 * math.log2(tp) + 12 * index
            values.append(LabeledHardwareTrial(name, config, hardware, target))
    return values


def test_paired_platform_holdouts_and_serialization():
    result = compare_hardware_heldout_models(trials(), ridge_lambda=0.01)
    assert [fold.held_out_platform for fold in result.folds] == [
        "H100-test", "H200-test", "H300-synthetic"
    ]
    assert result.test_samples == 12
    assert all(f.train_samples == 8 and f.test_samples == 4 for f in result.folds)
    assert all(f.config_rmse >= 0 and f.conditioned_rmse >= 0 for f in result.folds)
    assert result.to_dict()["metric"] == "throughput_per_s"
    assert result.rmse_reduction_pct is not None


def test_reordered_inputs_are_deterministic():
    forward = compare_hardware_heldout_models(trials(), ridge_lambda=0.01)
    backward = compare_hardware_heldout_models(reversed(trials()), ridge_lambda=0.01)
    assert forward.to_dict() == backward.to_dict()


def test_insufficient_platforms_rejected():
    with pytest.raises(ValueError, match="at least three platforms"):
        compare_hardware_heldout_models(trials()[:8])


def test_inconsistent_hardware_within_platform_rejected():
    values = trials()
    first = values[0]
    changed = HardwareFeatures(
        gpu_count=4,
        memory_gib_per_gpu=first.hardware.memory_gib_per_gpu,
        memory_bandwidth_gbps_per_gpu=first.hardware.memory_bandwidth_gbps_per_gpu,
        compute_tflops_per_gpu=first.hardware.compute_tflops_per_gpu,
        interconnect_bandwidth_gbps=first.hardware.interconnect_bandwidth_gbps,
    )
    values[0] = LabeledHardwareTrial(first.platform, first.config, changed, first.target)
    with pytest.raises(ValueError, match="multiple hardware descriptors"):
        compare_hardware_heldout_models(values)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="finite and positive"):
        compare_hardware_heldout_models(trials(), ridge_lambda=float("nan"))
    with pytest.raises(ValueError, match="at least four"):
        compare_hardware_heldout_models(trials(), min_train_samples=2)
    sample = trials()[0]
    with pytest.raises(ValueError, match="target must be finite"):
        LabeledHardwareTrial(sample.platform, sample.config, sample.hardware, math.nan)
    with pytest.raises(ValueError, match="platform must not be empty"):
        LabeledHardwareTrial("", sample.config, sample.hardware, 1.0)
