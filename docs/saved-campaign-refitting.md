# Saved campaign refitting research workflow

`evaluate_saved_refitting` connects saved benchmark campaigns to the
experimental leave-one-hardware-platform-out refitting bootstrap. It performs
no benchmark execution, GPU allocation, scheduling, or deployment.

```python
from adaptive_gpu_orchestrator.modeling import (
    evaluate_saved_refitting,
    write_saved_refitting_report,
)

report = evaluate_saved_refitting(
    campaigns,
    optimization_store=optimization_store,
    measured_hardware=measured_hardware,
    metric="throughput_per_s",
    iterations=200,
    confidence=0.95,
    seed=17,
)
write_saved_refitting_report(report, "research/refitting.json")
```

## Audit and validation steps

1. Reuse the exact matching-workload, measured-hardware, GPU-count and
   software-provenance checks from `compare_saved_campaigns`.
2. Audit saved trial campaign identities and software-field completeness
   with `audit_campaign_design`. Shared optimization IDs across campaigns
   cause a hard failure: possible evidence reuse.
3. Require at least two distinct campaigns per platform and three distinct
   hardware descriptors. Preserve campaign IDs while filtering failed trials
   and missing/nonfinite targets.
4. Perform leave-one-platform-out evaluation, resampling only training
   campaigns within their original hardware platform and refitting both
   models for every draw. Held-out test campaigns stay fixed.
5. Compare the refitting point estimate with the original paired model
   comparison, and fail if the results disagree.
6. Export a deterministic JSON containing the design audit, evidence
   comparison, and conditional refitting interval.

The report contains experiment identifiers and provenance: review sharing
and retention policies before distributing it.

**Limits:** This is conditional training-campaign sensitivity, NOT a
calibrated confidence interval for unseen GPU families. The held-out set
is reused in every draw, and GPU platforms are not resampled. Software
mismatch may only be allowed using an explicit argument; missing software
fields still require manual verification. Reproducibility depends on saved
input evidence and seed. The source experiment identity must still be
verified externally, and the synthetic tests are not H100/H200 evidence.
