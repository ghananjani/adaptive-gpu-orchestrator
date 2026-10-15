# DCGM + Time-Series Analysis

v0.8 moves workload telemetry from cluster averages toward per-GPU evidence.

## Per-GPU profile data

Each new telemetry sample keeps the existing aggregate fields and also stores one record per observed GPU:

- GPU index and UUID
- GPU utilization
- memory-copy utilization
- memory used/total and memory pressure
- temperature
- power draw and power limit

Old profiles remain valid. When a historical profile has no per-GPU records, v0.8 returns an explicit aggregate-only finding instead of failing.

## Straggler analysis

The analyzer computes a mean utilization per GPU across the profile.

For three or more GPUs, the reference is the median GPU utilization. For two GPUs, the reference is the higher mean utilization. A GPU is flagged when its mean utilization trails the reference by at least the configured threshold, defaulting to 15 percentage points.

This is a signal, not a root-cause claim.

## Phase alignment

Users can define workload windows using offsets relative to the profile start:

```text
NAME:START_SECONDS:END_SECONDS
```

Example:

```bash
--phase warmup:0:30 --phase steady:30:120
```

Each phase reports aggregate utilization, memory pressure, power, and per-GPU mean utilization for samples inside the window.

## DCGM Exporter ingestion

The parser accepts Prometheus exposition lines from DCGM Exporter and preserves all numeric `DCGM_*` fields.

Common operational fields include GPU utilization, memory-copy utilization, framebuffer used/free/total, power usage, GPU temperature, energy and NVLink counters.

Concatenated scrapes are converted into sequential frames by detecting repeated GPU/metric pairs. The sampling interval is supplied by the caller.

## dcgmi dmon ingestion

The parser also accepts tabular `dcgmi dmon` traces such as GPU activity, Tensor Core activity and DRAM activity.

Column names are preserved exactly so newer or custom field selections remain analyzable without a schema update.

## DCGM cross-check

When DCGM device utilization is present, the time-series report performs an independent utilization-imbalance check using the same configured straggler threshold.

The DCGM signal is treated as corroborating evidence, not as proof of causation.

## CLI

```bash
agoctl timeseries analyze \
  --profile <experiment-id> \
  --phase warmup:0:30 \
  --phase steady:30:120 \
  --dcgm-file dcgm.prom \
  --dcgm-format exporter
```

Saved reports live under:

```text
.ago/timeseries/
```

Inspect them with:

```bash
agoctl timeseries list
agoctl timeseries show <report-id>
```

## Research extensions

- timestamp alignment across telemetry, model phases and NCCL events
- DCGM profiling-field normalization
- NVLink/PCIe traffic time series
- repeated-run confidence intervals
- H100/H200 signature comparison
- anomaly detection for intermittent stragglers
- per-rank communication/compute overlap analysis
