# v1.7 Campaign-cluster bootstrap for held-out prediction errors

Use `bootstrap_campaign_rmse_reduction` with paired residuals obtained from
**held-out predictions**, not training predictions.

```python
from adaptive_gpu_orchestrator.modeling import (
    CampaignPairedResidual, bootstrap_campaign_rmse_reduction,
)

rows = [
    CampaignPairedResidual(platform, campaign_id, config_prediction - actual,
                           conditioned_prediction - actual)
    for platform, campaign_id, config_prediction, conditioned_prediction, actual
    in heldout_predictions
]
report = bootstrap_campaign_rmse_reduction(
    rows, iterations=2000, confidence=0.95, seed=17,
)
print(report.to_dict())
```

The estimator resamples **whole campaigns with replacement within platform**,
preserving the number of campaigns in each platform. Requests from the same
campaign are not resampled separately. At least two platforms and two
campaigns per platform are required. Inputs must carry unique campaign IDs
across platforms. Seeds, platform/campaign sorting and intra-campaign sorting
make results reproducible independent of input row order.

The point estimate is percentage reduction in pooled held-out RMSE:
`100 * (1 - conditioned_RMSE / baseline_RMSE)`. If baseline RMSE is zero,
the reduction is undefined (null). A negative percentage is a worsening,
not an improvement. Confidence bounds use basic empirical percentile
interpolation on the resampled cluster-reduction distribution.

**Interpretation limits:** These intervals describe variation conditional on
fixed supplied out-of-fold predictions and the observed hardware platforms.
The bootstrap does NOT refit either model or resample hardware platforms,
so intervals are not calibrated across unseen GPU platforms and can
understate total prediction uncertainty. The module does not compute p-values
or establish causal hardware advantage. Small campaign counts are inadequate
for publication-grade confidence claims; conduct independent campaigns,
more platform families, and a refitting hierarchical bootstrap where justified.

This module does not convert saved campaigns to residuals automatically:
capture paired predictions with campaign identifiers explicitly before
calling it. Never infer campaign identity from individual trial indices.
No benchmark execution, scheduling or GPU service mutations occur.
