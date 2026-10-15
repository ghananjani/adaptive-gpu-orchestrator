# Research Roadmap

This repository is both an engineering platform and a substrate for ML-systems research.

## R1 — Communication-aware placement

**Question:** Can a workload communication graph and a measured hardware topology graph predict better rank placement than scheduler-local heuristics?

Inputs:
- measured NVLink/NVSwitch/PCIe/IB bandwidth and latency
- collective profile by rank
- TP/PP/DP structure

Evaluation:
- tokens/sec or step time
- collective time fraction
- scaling efficiency
- placement-search overhead

## R2 — Adaptive parallelism

**Question:** Can short canary runs identify a near-optimal TP/PP/DP configuration without exhaustively benchmarking the space?

Potential methods:
- successive halving
- Bayesian optimization
- learned cost models
- topology-aware pruning

## R3 — Heterogeneous accelerator scheduling

**Question:** How should workloads be placed when H100, H200 and future accelerators coexist?

Research dimensions:
- memory capacity/bandwidth
- compute capability
- interconnect asymmetry
- energy efficiency
- fairness / queueing constraints

## R4 — KV-cache-aware orchestration

**Question:** Can real-time KV-cache pressure drive admission control, replica scaling and request placement better than GPU-utilization-only policies?

Metrics:
- TTFT
- inter-token latency
- cache hit rate
- preemption/eviction rate
- effective concurrency
- throughput per GB HBM

## R5 — SLO-aware energy optimization

**Question:** What is the minimum energy operating point that still satisfies p95/p99 inference SLOs?

Controls:
- GPU power caps
- replica count
- batch size
- prefill/decode allocation
- heterogeneous placement

## R6 — Failure and degradation detection

**Question:** Can deviations from historical communication and telemetry baselines identify degraded GPUs, links or nodes before application failure?

Signals:
- XID/ECC
- link errors
- NCCL bandwidth drift
- per-rank step-time drift
- temperature/power anomalies

## Dataset schema

Every experiment should eventually record:

```text
hardware identity
topology
software versions
model + revision
sequence/input distribution
batch/concurrency
TP/PP/DP
placement
runtime flags
throughput
TTFT / ITL / p50 / p95 / p99
VRAM / HBM bandwidth
collective bandwidth/time
power / joules-per-token
failure / retry metadata
```

This dataset becomes the foundation for the learned predictor in later versions.
