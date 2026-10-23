import pytest

from adaptive_gpu_orchestrator.modeling.hardware import HardwareFeatures
from adaptive_gpu_orchestrator.modeling.refitting import (
    CampaignHardwareTrial,
    refit_campaign_bootstrap,
)
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig


def observations():
    trials = []
    for index, platform in enumerate(("GPU-A", "GPU-B", "GPU-C")):
        hw = HardwareFeatures(8, 64 + 16 * index, 1200 + 300 * index,
                              40 + 10 * index, 250 + 50 * index)
        for campaign in range(3):
            for tp in (1, 2, 4):
                trials.append(CampaignHardwareTrial(
                    platform, f"{platform}-{campaign}",
                    ParallelismConfig(tensor_parallel=tp), hw,
                    float(70 + 15 * index + 10 * tp + campaign),
                ))
    return trials


def test_refitting_is_repeatable_and_order_independent():
    a = refit_campaign_bootstrap(observations(), iterations=20, seed=42)
    b = refit_campaign_bootstrap(reversed(observations()), iterations=20, seed=42)
    assert a.to_dict() == b.to_dict()
    assert a.test_samples == 27
    assert a.campaign_count == 9
    assert a.platform_count == 3
    assert a.valid_iterations == 20
    assert a.lower_pct <= a.upper_pct


def test_inadequate_campaign_replication_rejected():
    only_first = [row for row in observations() if row.campaign_id.endswith("-0")]
    with pytest.raises(ValueError, match="at least two campaigns"):
        refit_campaign_bootstrap(only_first, iterations=20)


def test_reused_campaign_id_across_platforms_rejected():
    trials = observations()
    original = trials[0]
    trials.append(CampaignHardwareTrial(
        "GPU-B", original.campaign_id, original.config, trials[-1].hardware, 2.0
    ))
    with pytest.raises(ValueError, match="spans multiple"):
        refit_campaign_bootstrap(trials, iterations=20)


def test_missing_hardware_diversity_rejected():
    trials = observations()
    same = trials[0].hardware
    inputs = [CampaignHardwareTrial(r.platform, r.campaign_id, r.config,
                                    same, r.target) for r in trials]
    with pytest.raises(ValueError, match="distinct hardware"):
        refit_campaign_bootstrap(inputs, iterations=20)


@pytest.mark.parametrize("params", [
    {"iterations": 0}, {"confidence": 1.0}, {"confidence": float("nan")},
    {"ridge_lambda": -1.0},
])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        refit_campaign_bootstrap(observations(), **params)
