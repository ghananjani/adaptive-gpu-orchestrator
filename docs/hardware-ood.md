# v1.7 measured hardware features and OOD diagnostics

This increment introduces an **explicit, supplied, measured** hardware descriptor
and a conservative out-of-distribution check. It does not substitute vendor SKU
tables, extrapolate missing measurements, or change the v1.6 persisted model schema.

## API

```python
from adaptive_gpu_orchestrator.modeling import HardwareFeatures, assess_hardware_ood

reference = HardwareFeatures(
    gpu_count=8,
    memory_gib_per_gpu=80,
    memory_bandwidth_gbps_per_gpu=2000,
    compute_tflops_per_gpu=50,
    interconnect_bandwidth_gbps=400,
)
# Values above are illustration-only inputs, NOT verified SKU specifications.
candidate = HardwareFeatures(
    gpu_count=8,
    memory_gib_per_gpu=96,
    memory_bandwidth_gbps_per_gpu=2000,
    compute_tflops_per_gpu=50,
    interconnect_bandwidth_gbps=400,
)
report = assess_hardware_ood([reference], candidate)
print(report.to_dict())
```

All five input values are mandatory, positive and finite; GPU count must be
an integer. The user is responsible for consistent measurement definitions.
In particular, accelerator FLOPS depend strongly on dtype/sparsity; do not mix
BF16 dense and FP8 sparse figures. Interconnect bandwidth must refer to the
same direction, link/group accounting, and topology measurement.

Distance uses standardized log2 feature vectors, comparing candidates with
the nearest reference observation; the standardization scale has a floor of
one log2 unit to keep constant features numerically stable. A candidate is
flagged if its nearest distance exceeds the requested threshold **or** any
feature is outside the observed min/max range. An exact observed configuration
scores zero.

This is a descriptive screening heuristic, **not** a predictive interval, a
calibrated probability, or proof of workload transferability. Same-feature
hardware may still differ in NCCL implementation, topology, firmware, driver,
model architecture, or workload behavior. Do not use this output for automatic
scheduler deployment.

Combine with `evaluate_hardware_transfer` to report empirical held-out
platform errors before trying a hardware-conditioned performance predictor.
