# Paired hardware-held-out model comparison

This experiment compares the existing configuration-only ridge estimator and the
experimental additive hardware-conditioned ridge estimator on **identical**
held-out hardware platforms, with no test-platform trials used for fitting.

## Use

```python
from adaptive_gpu_orchestrator.modeling import (
    LabeledHardwareTrial, compare_hardware_heldout_models,
)

# Construct these observations from real matched campaign exports and
# separately measured hardware descriptors. No SKU values are inferred.
observations = [
    LabeledHardwareTrial(platform, config, hardware, measured_throughput)
    for platform, config, hardware, measured_throughput in input_records
]
result = compare_hardware_heldout_models(
    observations, metric="throughput_per_s", ridge_lambda=1.0,
)
print(result.to_dict())
```

The library API requires **at least three distinct platforms**, with a single
consistent hardware descriptor per platform. The conditioned estimator needs
at least two training hardware descriptors when the third platform is held
out. Four or more training trials and two distinct TP/PP/DP/batch/concurrency
configurations are required for every fold.

For each hardware-held-out fold, report training and test counts, config-only
RMSE and MAE, conditioned RMSE and MAE, and relative RMSE reduction. Aggregate
metrics are pooled over trial-level held-out residuals (rather than averaging
fold metrics). Positive reduction means lower held-out RMSE; negative means
worse predictions. Zero baseline RMSE produces a null relative reduction.

**OOD research exception:** the held-out hardware is necessarily unseen by
the conditioned model. The evaluator explicitly permits prediction only to
measure the error of cross-platform extrapolation. The normal conditioned
model's out-of-range rejection remains active. This exception is not
authorization for scheduling, model deployment or benchmark execution.

The tests use **synthetic** hardware descriptors and throughput surfaces;
they are not evidence of H100/H200 performance gains. Compare on real
matched model sizes, precisions, ISL/OSL, software and topology. Include
multiple campaigns per GPU family and evaluate campaign-level leakage
separately. The reduction metric is descriptive, without confidence
intervals or calibrated uncertainty. The platform labels are supplied by
the caller; no GPU inventory or physical testing is automatically done.

This increment does not mutate saved v1.6 models or GPU/scheduler state.
