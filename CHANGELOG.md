# Changelog

## 0.5.0a0

Guarded Validation milestone:

- baseline-vs-candidate promotion validation
- throughput-gain guardrails
- p99-latency, energy and memory-pressure regression budgets
- strict missing-metric handling for configured guardrails
- accept/reject decision model
- rollback recommendation on rejected candidates
- persisted validation audit history
- deployment-plan diff utility
- `agoctl validate`
- `agoctl validations list/show`
- validation, persistence, CLI and plan-diff tests

## 0.4.0a0

Cluster Controller foundation:

- deployment plan and policy models
- recommend-only safety default
- GPU-budget and exact-allocation policy validation
- topology-aware rank placement reuse
- Kubernetes `batch/v1 Job` renderer
- Slurm script renderer
- corrected Slurm directive ordering
- node selector and partition support
- `agoctl plan`
- planning from direct configuration or saved optimizer winner
- controller, renderer and CLI tests

## 0.3.0a0

Configuration Optimizer milestone:

- reproducible TP/PP/DP + batch/concurrency configuration model
- GPU-budget-aware candidate generation
- exact-world-size search mode
- command-based canary benchmark runner
- benchmark environment contract via AGO_* variables
- throughput, latency, energy and balanced objectives
- p99-latency, energy, minimum-throughput and memory-pressure constraints
- persisted optimization history
- `agoctl optimize`
- `agoctl optimizations list/show`
- dry-run search-space inspection
- deterministic optimizer and CLI tests

## 0.2.0a0

Workload Intelligence milestone:

- continuous telemetry sampling
- optional NVML provider
- workload profiling around arbitrary commands
- experiment persistence and history
- workload-level bottleneck diagnosis
- `agoctl profile`
- `agoctl experiments list/show`
- `agoctl doctor --profile`
- public profiling Python API
- deterministic no-GPU tests for profiling and storage

## 0.1.0a0

Initial pre-alpha GPU Doctor milestone:

- NVIDIA GPU discovery
- GPU topology parsing
- point-in-time utilization / memory / thermal / power capture
- NCCL benchmark parsing and execution wrapper
- explainable bottleneck diagnosis
- topology-aware rank placement baseline
- CLI and JSON output
- no-GPU CI/test support
