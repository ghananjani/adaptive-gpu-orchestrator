import pytest

from adaptive_gpu_orchestrator.modeling.conditioned import (
    ConditionedRidgeModel,
    HardwareTrainingSample,
    fit_hardware_conditioned_model,
)
from adaptive_gpu_orchestrator.modeling.hardware import HardwareFeatures
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig


def hardware(memory):
    return HardwareFeatures(
        gpu_count=8,
        memory_gib_per_gpu=memory,
        memory_bandwidth_gbps_per_gpu=2000,
        compute_tflops_per_gpu=60,
        interconnect_bandwidth_gbps=400,
    )


def samples():
    observations = []
    for memory, bonus in ((80, 0), (96, 10)):
        for tp in (1, 2, 4, 8):
            observations.append(HardwareTrainingSample(
                ParallelismConfig(tensor_parallel=tp),
                hardware(memory),
                100.0 + 20.0 * (tp.bit_length() - 1) + bonus,
            ))
    return observations


def test_conditioned_fit_and_roundtrip():
    model = fit_hardware_conditioned_model(samples(), ridge_lambda=1e-6)
    assert model.fit_rmse < 0.01
    assert model.sample_count == 8
    assert model.predict(ParallelismConfig(tensor_parallel=4), hardware(96)) == pytest.approx(
        150, abs=0.01
    )
    loaded = ConditionedRidgeModel.from_dict(model.to_dict())
    assert loaded.to_dict() == model.to_dict()
    assert loaded.predict(ParallelismConfig(tensor_parallel=2), hardware(80)) == pytest.approx(
        120, abs=0.01
    )


def test_ood_guard_blocks_unmeasured_hardware():
    model = fit_hardware_conditioned_model(samples())
    with pytest.raises(ValueError, match="outside training distribution"):
        model.predict(ParallelismConfig(), hardware(120))
    assert isinstance(model.predict(ParallelismConfig(), hardware(120), allow_ood=True), float)


def test_input_order_is_deterministic():
    first = fit_hardware_conditioned_model(samples(), ridge_lambda=1e-5)
    second = fit_hardware_conditioned_model(list(reversed(samples())), ridge_lambda=1e-5)
    assert first.to_dict() == second.to_dict()


def test_rejects_insufficient_platform_diversity_and_samples():
    with pytest.raises(ValueError, match="at least four"):
        fit_hardware_conditioned_model(samples()[:3])
    with pytest.raises(ValueError, match="distinct hardware"):
        fit_hardware_conditioned_model(samples()[:4])
    with pytest.raises(ValueError, match="finite and positive"):
        fit_hardware_conditioned_model(samples(), ridge_lambda=float("nan"))


def test_corrupt_model_schema_rejected():
    model = fit_hardware_conditioned_model(samples())
    data = model.to_dict()
    data["feature_names"] = []
    with pytest.raises(ValueError, match="feature schema"):
        ConditionedRidgeModel.from_dict(data)
