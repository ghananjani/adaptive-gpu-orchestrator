# Saved Campaign Evidence → Paired Hardware Comparison

This API runs the existing paired hardware-held-out comparison directly on saved
`CampaignReport` records and their `OptimizationStore` trial evidence. It never
launches benchmarks, schedules jobs, modifies optimizer stores or deploys models.

## Strict evidence contract

Each campaign must have nonempty `spec.tags` fields:
`model`, `workload`, `precision`, `input_tokens`, `output_tokens`
and `benchmark_protocol`. All six values must match exactly between campaigns.
These tags describe the experiment protocol, not a guarantee of identical runtime
behavior; record additional workload constraints independently.

Hardware descriptors are **operator-measured inputs**, keyed by the exact
`gpu_models` signature emitted in `campaign_trial_rows`. GPU count must
match the saved cluster snapshot. This adapter never infers GPU bandwidth,
compute FLOPS, memory capacity or interconnect properties from model names.

CUDA runtime/toolkit, NCCL version and container image digest are compared
across platforms and must match by default. `allow_software_mismatch=True`
explicitly allows cross-software comparisons with a warning in the report.
Contradictory provenance within the same platform always raises an error.
Missing software version values do not prove matching environments; authors
should verify completeness before using results in publications.

## Python example

```python
from adaptive_gpu_orchestrator.campaigns import CampaignStore
from adaptive_gpu_orchestrator.optimizer import OptimizationStore
from adaptive_gpu_orchestrator.modeling import (
    HardwareFeatures, compare_saved_campaigns, write_campaign_comparison,
)

campaign_store = CampaignStore()
optimization_store = OptimizationStore()
campaigns = [campaign_store.load(item) for item in campaign_ids]

hardware = {
    # Exact platform signature -> independently measured features
    # "NVIDIA H100 80GB HBM3": HardwareFeatures(...),
}
report = compare_saved_campaigns(
    campaigns,
    optimization_store=optimization_store,
    measured_hardware=hardware,
    metric="throughput_per_s",
)
write_campaign_comparison(
    report,
    json_path="results/paired-evidence.json",
    csv_path="results/paired-folds.csv",
)
```

The canonical JSON contains selected campaign IDs, workload tags, measured
hardware descriptors, software provenance per platform, exclusion counts,
trial count, SHA-256 digest over included trial evidence, warnings, and
the original paired held-out metrics. The CSV has per-platform comparison
errors. The SHA-256 digest is a content identifier for normalized included
trial rows, **not** a signature, source authenticity proof or complete
fingerprint of all provenance metadata.

Failed trials and nonfinite/missing target metrics are excluded and counted.
There must be enough remaining successful evidence for at least three
hardware-distinct held-out platform folds. The comparator explicitly bypasses
the hardware OOD guard only for offline held-out error measurement, not for
deployment or automatic recommendations.

## Scientific cautions and next research work

Matching six protocol tags is necessary but not sufficient: verify model
revision, tokenizer, sampling, batch/concurrency schedules, power limits,
topology, kernel versions, prompt mix, cache state and repeats. Software
versions should be complete and independently audited. Results on synthetic
unit fixtures are *not* real H100/H200 evidence. The interface makes real
campaign evaluation possible once those artifacts are supplied.

The current fold definition is hardware signature, **not** campaign or
replicate. Multiple campaigns from one platform can induce correlated
residuals. Plan leave-one-campaign-out sensitivity analyses, hierarchical
confidence intervals and workload stratification before claiming predictive
superiority or statistical significance.
