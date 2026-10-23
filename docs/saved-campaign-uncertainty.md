# Saved-campaign held-out uncertainty workflow

The experimental `evaluate_saved_campaign_uncertainty` API connects persisted
`CampaignReport` and `OptimizationStore` records to paired
leave-one-platform-out predictions and then a seeded campaign-cluster bootstrap.

```python
from adaptive_gpu_orchestrator.modeling import evaluate_saved_campaign_uncertainty

report = evaluate_saved_campaign_uncertainty(
    campaigns,
    optimization_store=optimization_store,
    measured_hardware=measured_hardware,
    metric="throughput_per_s",
    iterations=2000,
    confidence=0.95,
    seed=17,
)
print(report.to_dict())
```

The function first calls `compare_saved_campaigns` to apply the established
workload-tag, GPU-count and software-provenance safeguards. It extracts the
same valid stored trial observations, fits the baseline and hardware-conditioned
models on all *other* platforms, and calculates each held-out trial's paired
prediction residuals, retaining the **campaign ID** for cluster resampling.
It checks per-platform RMSE against the existing comparison evaluator before
returning, failing closed if the results diverge.

The bootstrap resamples whole campaigns within each observed platform. At
least two separate campaigns per platform are required. Passing repeated
trial observations from only one campaign per platform is insufficient.

The output includes original comparison evidence, per-trial paired residuals
and the bootstrap interval. Residuals expose campaign and platform IDs; treat
reports as potentially sensitive benchmark artifacts.

**Research limitations:** The OOD bypass is explicitly restricted to offline
holdout scoring. The bootstrap treats predictions as fixed, does not refit
the models for every resample and does not sample new GPU families. This is
NOT a calibrated interval for transfer performance on unobserved hardware,
and a confidence interval excluding zero is not alone proof of model
superiority. Campaign tags and software provenance still require independent
verification against experimental protocol. No GPU jobs or deployments run.

For fully defensible uncertainty inference, gather independent campaigns
per hardware platform, verify workload/precision equivalence and implement
nested refitting resampling or a platform-level hierarchical model.
