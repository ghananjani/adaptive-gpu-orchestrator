# Experimental hardware-conditioned ridge model

The additive, dependency-free hardware-conditioned model is an opt-in research API.
It does **not** change the v1.6 performance-model JSON schema, CLI, or deployment
controller. Training data must use comparable workloads, model precision, and
measurement definitions across all sampled platforms.

```python
from adaptive_gpu_orchestrator.modeling import (
    HardwareFeatures, HardwareTrainingSample, fit_hardware_conditioned_model,
)
from adaptive_gpu_orchestrator.optimizer import ParallelismConfig

# Supply real, consistently measured descriptors and benchmark targets.
measurements = [
    HardwareTrainingSample(config, measured_hardware, throughput)
    for config, measured_hardware, throughput in observations
]
model = fit_hardware_conditioned_model(measurements, metric="throughput_per_s")
prediction = model.predict(config, measured_hardware)
model_json = model.to_dict()  # Dedicated schema; not a v1.6 model report.
```

The estimator concatenates log2 TP/PP/DP/batch/concurrency with log2 measured
GPU count, memory capacity, memory bandwidth, compute throughput, and
interconnect bandwidth; standardizes the features; and fits additive ridge
regression. It requires at least four samples and two distinct measured
hardware descriptors.

**Safety gate:** predictions on hardware outside the observed component-wise
feature range are rejected by default. `allow_ood=True` is an explicit
research-only bypass and is not permission to deploy. The distance/range check
is descriptive and not a calibrated uncertainty score.

Limitations: the additive model cannot learn architecture-specific interactions
or guarantees of H100/H200 transfer. The training RMSE is descriptive, not a
generalization estimate. Evaluate leave-one-hardware-platform-out performance
using the existing `evaluate_hardware_transfer` facility and compare it with
the config-only baseline. No claimed gains without real held-out benchmarks.
