import pytest

from adaptive_gpu_orchestrator.modeling.bootstrap import (
    CampaignPairedResidual,
    bootstrap_campaign_rmse_reduction,
)


def observations():
    return [
        CampaignPairedResidual(platform, f"{platform}-{campaign}", error, error * 0.8)
        for platform in ("H100", "H200", "L40S")
        for campaign in (1, 2, 3)
        for error in (1.0 + campaign, 2.0 + campaign)
    ]


def test_fixed_seed_and_permutation_independent():
    rows = observations()
    forward = bootstrap_campaign_rmse_reduction(rows, iterations=150, seed=17)
    reverse = bootstrap_campaign_rmse_reduction(reversed(rows), iterations=150, seed=17)
    assert forward.to_dict() == reverse.to_dict()
    assert forward.estimate_pct == pytest.approx(20.0)
    assert forward.lower_pct == pytest.approx(20.0)
    assert forward.upper_pct == pytest.approx(20.0)
    assert forward.valid_iterations == 150


def test_negative_reduction_is_not_suppressed():
    rows = [
        CampaignPairedResidual(row.platform, row.campaign_id, row.config_error,
                               row.conditioned_error * 2)
        for row in observations()
    ]
    result = bootstrap_campaign_rmse_reduction(rows, iterations=100)
    assert result.estimate_pct < 0
    assert result.upper_pct < 0


def test_requires_campaign_replication_per_platform():
    rows = [r for r in observations() if r.campaign_id.endswith("-1")]
    with pytest.raises(ValueError, match="independent campaigns"):
        bootstrap_campaign_rmse_reduction(rows)


def test_rejects_cross_platform_campaign_reuse():
    rows = observations()
    rows.append(CampaignPairedResidual("H200", "H100-1", 1, 1))
    with pytest.raises(ValueError, match="more than one platform"):
        bootstrap_campaign_rmse_reduction(rows)


def test_zero_baseline_has_no_defined_reduction():
    rows = [CampaignPairedResidual(r.platform, r.campaign_id, 0, 1)
            for r in observations()]
    result = bootstrap_campaign_rmse_reduction(rows, iterations=100)
    assert result.estimate_pct is None
    assert result.lower_pct is None
    assert result.valid_iterations == 0


@pytest.mark.parametrize("kwargs", [
    {"iterations": 0}, {"confidence": 1.0}, {"confidence": float("nan")},

])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        bootstrap_campaign_rmse_reduction(observations(), **kwargs)
