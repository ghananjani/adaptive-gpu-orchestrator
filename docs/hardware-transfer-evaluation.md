# v1.7 research increment: cross-hardware holdout evaluation

This is the first **evaluation** component for hardware-aware research, not yet
a hardware-aware predictor. Existing v1.6 model schema and CLI remain unchanged.

## Purpose

Test whether a configuration-only ridge model fitted on one GPU platform
generalizes to a *different* platform. The correct split is by hardware identity,
not a random split of trials (which would leak platform-specific effects).

`evaluate_hardware_transfer(rows, metric="throughput_per_s")` accepts iterable
dictionaries from the existing `campaign_trial_rows(...)` export. It groups by
`gpu_models` by default, holds out each distinct platform in turn, fits the
existing v1.6 ridge estimator on the other platforms, and evaluates on the
held-out platform.

Each fold records train/test counts, training platform identifiers, RMSE,
MAE, and a training-mean baseline RMSE. The aggregate reports sample-weighted
pooled RMSE/MAE across held-out predictions.

```python
from adaptive_gpu_orchestrator.modeling import evaluate_hardware_transfer
from adaptive_gpu_orchestrator.campaigns import campaign_trial_rows

rows = []
for campaign in campaigns:
    rows.extend(campaign_trial_rows(campaign, optimization_store=optimization_store))
report = evaluate_hardware_transfer(rows, metric="throughput_per_s")
print(report.to_dict())
```

## Research limitations

- **No hardware features are used by the predictor.** Platform identity only
  determines holdout splits. This is not yet transfer learning.
- Distinct `gpu_models` identities such as H100 and H200 must be available.
  Hardware/software provenance differences *within* a GPU family may remain
  important confounders.
- Missing/nonfinite target values and failed trials are excluded.
- A fold must have enough training samples and at least two distinct configs.
- Errors are in original metric units, not normalized percentages.
- Different workloads, driver/software versions and search-space coverage can
  confound cross-platform error. Keep other conditions matched before making
  scientific comparisons.
- A model trained on H100 does not become trustworthy on H200 because a
  numerical estimate can be produced. Report holdout errors and baselines.
- No GPU jobs, scheduler changes, or deployed inference paths are invoked.

## Next work

Add validated hardware attributes (memory bandwidth, accelerator capacity,
interconnect topology and communication baseline) as explicit features;
implement group-aware validation and out-of-distribution checks before
offering hardware-conditioned recommendations.
