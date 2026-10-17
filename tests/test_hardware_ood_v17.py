import pytest

from adaptive_gpu_orchestrator.modeling.hardware import (
    HardwareFeatures,
    assess_hardware_ood,
)


def sample(**changes):
    values = {
        "gpu_count": 8,
        "memory_gib_per_gpu": 80.0,
        "memory_bandwidth_gbps_per_gpu": 2000.0,
        "compute_tflops_per_gpu": 50.0,
        "interconnect_bandwidth_gbps": 400.0,
    }
    values.update(changes)
    return HardwareFeatures(**values)


def test_hardware_roundtrip_and_order():
    hardware = sample()
    assert HardwareFeatures.from_dict(hardware.to_dict()) == hardware
    assert len(hardware.vector()) == 5


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_measurements_rejected(value):
    with pytest.raises((TypeError, ValueError), match="finite positive"):
        sample(memory_bandwidth_gbps_per_gpu=value)


def test_gpu_count_must_be_integer():
    with pytest.raises(TypeError, match="integer"):
        sample(gpu_count=2.5)


def test_missing_fields_rejected():
    with pytest.raises(ValueError, match="all measured"):
        HardwareFeatures.from_dict({"gpu_count": 8})


def test_exact_observation_is_in_distribution():
    a = sample()
    result = assess_hardware_ood([a, a], a)
    assert result.distance == pytest.approx(0.0)
    assert result.out_of_distribution is False
    assert result.outside_observed_range == ()
    assert result.reference_samples == 2
    assert result.to_dict()["out_of_distribution"] is False


def test_unseen_hardware_flagged_even_if_nearest_distance_small():
    a = sample()
    result = assess_hardware_ood(
        [a], sample(memory_gib_per_gpu=96.0), threshold=20
    )
    assert result.distance < 20
    assert result.out_of_distribution
    assert result.outside_observed_range == ("memory_gib_per_gpu",)


def test_nearest_distance_and_input_order_independent():
    a, b = sample(), sample(gpu_count=4)
    candidate = sample(gpu_count=8)
    forward = assess_hardware_ood([a, b], candidate)
    reverse = assess_hardware_ood([b, a], candidate)
    assert forward.to_dict() == reverse.to_dict()
    assert forward.distance == 0
    assert not forward.out_of_distribution


def test_threshold_validation_and_empty_reference():
    with pytest.raises(ValueError, match="threshold"):
        assess_hardware_ood([sample()], sample(), threshold=float("nan"))
    with pytest.raises(ValueError, match="at least one"):
        assess_hardware_ood([], sample())
