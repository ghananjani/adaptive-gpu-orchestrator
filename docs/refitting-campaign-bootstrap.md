# Refitting-based campaign bootstrap (v1.7 experimental)

The `refit_campaign_bootstrap` API estimates sensitivity of **paired
leave-one-hardware-platform-out** prediction error comparisons to sampling
of independent training campaigns. It is a standalone read-only API;
saved campaign objects can be transformed to `CampaignHardwareTrial` after
strict provenance and workload checks.

For each fixed held-out hardware platform, each bootstrap draw samples
whole training campaigns with replacement **within each other platform**,
refits both models on those sampled campaigns, and scores the same fixed
held-out observations. It pools paired squared errors and calculates
`100 * (1 - conditioned_RMSE / configuration_RMSE)`.

This differs from the earlier fixed-residual bootstrap because fitted
coefficients change with every draw. The held-out tests never enter the
training data for their platform. The input requires at least three distinct
hardware descriptors and two campaigns per platform; campaign IDs must not
be reused across platforms. A deterministic random seed reproduces results
even if input order is reversed.

Example:

```python
from adaptive_gpu_orchestrator.modeling import (
    CampaignHardwareTrial, refit_campaign_bootstrap,
)

trials = [
    CampaignHardwareTrial(platform, campaign_id, config, measured_hardware, throughput)
    for platform, campaign_id, config, measured_hardware, throughput in observations
]
report = refit_campaign_bootstrap(trials, iterations=500, seed=17)
print(report.to_dict())
```

**Statistical interpretation:** Intervals are percentile summaries of model
sensitivity to resampled **training** campaigns. The same held-out campaigns
and hardware platforms are used on every draw; this is NOT a full hierarchical
bootstrap, not calibrated transfer uncertainty for an unobserved GPU family,
and not a causal significance test. A result excluding zero does not establish
hardware-conditioned superiority. Low cluster counts, confounded training
protocols and repeated use of the same held-out observations can severely
limit interpretation. Models are fit purely in memory; no GPU, scheduler,
container or production state is touched.
