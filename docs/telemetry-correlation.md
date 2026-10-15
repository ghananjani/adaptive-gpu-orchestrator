# Telemetry Correlation

v0.7 connects optimization results to the telemetry and communication evidence collected around a workload.

## Why offline correlation

Correlation is deliberately performed after an optimization run instead of sampling inside the canary runner.

This avoids:

- changing benchmark timing behavior
- introducing sampler threads into optimizer trials
- coupling workload execution to a specific telemetry provider
- making scheduler mutation a prerequisite for analysis

The inputs remain independently reproducible artifacts:

- a saved optimization result
- a saved workload profile
- optional NCCL benchmark output

## Correlation report

A report records:

- optimization ID
- workload-profile ID
- selected parallelism configuration
- normalized trial metrics
- adapter-specific metrics
- telemetry summary
- NCCL summary
- descriptive findings

Reports are stored under:

```text
.ago/correlations/
```

## Telemetry signals

The workload profile contributes:

- mean GPU utilization
- GPU-utilization standard deviation
- mean memory pressure
- maximum temperature
- mean power draw
- profile duration
- sample count
- maximum observed GPU count

When average power and profile duration are available, v0.7 estimates energy as:

```text
estimated_energy_joules = power_draw_mean_w × duration_s
```

When normalized throughput is also available, the report derives a generic energy-per-throughput-unit value. The throughput unit depends on the workload adapter, so this value must only be compared across runs using the same workload semantics.

## NCCL summary

Saved nccl-tests output can be supplied with `--nccl-output`.

The report summarizes:

- sample count
- mean/min/max bus bandwidth
- mean/max algorithm bandwidth
- largest tested message size
- bus bandwidth at the largest tested message size

NCCL values are a communication baseline, not a direct measurement of application collective utilization.

## Findings

Current findings flag signals such as:

- low mean GPU utilization
- high utilization imbalance
- high memory pressure
- high temperature
- communication-heavy distributed-training phases

If a PyTorch adapter reports a high communication fraction and NCCL data is present, the report places both observations together so topology and collective efficiency can be investigated.

These findings are intentionally non-causal. They identify evidence worth investigating; they do not assert root cause.

## CLI

```bash
agoctl correlate \
  --optimization <optimization-id> \
  --profile <experiment-id> \
  --nccl-output nccl.txt
```

Without NCCL data:

```bash
agoctl correlate \
  --optimization <optimization-id> \
  --profile <experiment-id>
```

Inspect history:

```bash
agoctl correlations list
agoctl correlations show <correlation-id>
```

## Next research extensions

- automatic time-alignment between workload phases and telemetry samples
- DCGM field ingestion
- per-GPU instead of cluster-average correlation
- topology-aware NCCL efficiency baselines
- regression models linking configuration choices to telemetry signatures
- confidence intervals across repeated canary runs
