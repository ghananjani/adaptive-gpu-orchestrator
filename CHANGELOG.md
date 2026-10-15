# Changelog

## 0.9.0a0

Statistical Experiment Studies milestone:

- repeated optimization-run grouping
- Student-t 95% confidence intervals
- baseline-vs-candidate absolute and percent change
- direction-aware relative improvement
- Cohen's d effect size
- selected-configuration stability
- adapter-specific metric aggregation
- throughput/latency/energy Pareto frontier
- persisted study reports
- `agoctl study analyze`
- `agoctl studies list/show`
- deterministic statistics, persistence and CLI tests

## 0.8.0a0

DCGM + Time-Series Analysis milestone:

- per-GPU telemetry persisted in new workload profiles
- backward-compatible historical profile loading
- DCGM Exporter Prometheus parser
- `dcgmi dmon` parser
- unknown numeric DCGM metric preservation
- per-GPU utilization/memory/power/thermal summaries
- persistent utilization-straggler detection
- phase-window alignment by profile offset
- optional DCGM utilization cross-checks
- persisted time-series reports
- `agoctl timeseries analyze|list|show`
- deterministic telemetry, DCGM, analysis, persistence and CLI tests

## 0.7.0a0

Telemetry Correlation milestone:

- correlation report model and persistence
- optimizer winner + workload-profile correlation
- optional NCCL baseline summarization
- utilization, imbalance, memory-pressure and thermal signals
- estimated energy from average power and profile duration
- generic energy-per-throughput-unit derivation
- adapter-native metric carry-through
- descriptive communication findings
- `agoctl correlate`
- `agoctl correlations list/show`
- correlation, store and CLI tests

## 0.6.0a0

Workload Adapters milestone:

- adapter protocol for native workload metrics
- vLLM serving metrics adapter
- distributed PyTorch training metrics adapter
- adapted command canary runner
- optimizer `--adapter generic|vllm|pytorch-distributed`
- native TTFT / TPOT / ITL preservation
- distributed communication / compute / phase timing preservation
- adapter-specific metrics stored in `TrialMetrics.extras`
- adapter metrics preserved through optimization persistence
- deterministic parser, runner and CLI tests

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
