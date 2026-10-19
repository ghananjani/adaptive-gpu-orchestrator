# Changelog

## Unreleased — v1.7 cross-hardware research foundations

- opt-in hardware-conditioned log-linear ridge prototype with explicit feature schema
- OOD guard blocks unmeasured hardware unless explicitly overridden
- separate serializable conditioned model, leaving legacy v1.6 schemas intact
- deterministic no-GPU tests and research limitations documentation

- explicit user-supplied measured hardware feature descriptors
- conservative log-feature out-of-distribution diagnostic and range guard
- new deterministic input-validation and OOD tests
- no SKU numbers inferred, no calibrated uncertainty claims, no scheduler changes

- leave-one-hardware-platform-out model evaluation on exported campaign trials
- per-platform RMSE / MAE and training-mean baseline RMSE
- pooled cross-platform error and deterministic research report serialization
- explicit separation of evaluation from hardware-conditioned prediction
- no changes to persisted v1.6 performance-model schema or deployment behavior


## 1.6.0a0

Performance Modeling + Experiment Recommendation milestone:

- dependency-free log-linear ridge performance model
- TP / PP / DP / batch / concurrency feature space
- train from every successful saved campaign trial
- default throughput, p99 latency, energy and memory-pressure targets
- explicit custom metric support
- per-metric fit RMSE / MAE / R²
- deterministic leave-one-out CV RMSE / MAE
- hardware/software provenance signature guard
- explicit mixed-hardware override
- persisted performance models under `.ago/models`
- recommend-only next-experiment ranking
- configurable exploit/explore acquisition weighting
- measured configurations excluded by default
- exploration based on standardized configuration-space distance
- no claim of calibrated predictive uncertainty
- `agoctl model train|list|show|recommend`
- no GPU, scheduler or benchmark mutation from model recommendation
- deterministic modeling, provenance-guard, persistence and CLI tests

## 1.5.0a0

Campaign Baselines + Research Export milestone:

- optional campaign-level NCCL all-reduce baseline
- saved nccl-tests output ingestion
- direct nccl-tests execution before campaign replicates
- raw NCCL samples plus mean/peak algorithm and bus bandwidth
- explicit CUDA toolkit provenance
- explicit/detected NCCL version provenance
- optional container image and digest provenance
- richer provenance included in resume compatibility checks
- trial-level campaign flattening
- CSV research export
- JSONL research export
- adapter-native `extra_*` metric columns
- NCCL/provenance fields included in exported rows
- `agoctl campaign export`
- deterministic baseline, provenance, export and CLI tests
- existing campaign manifest schema remains unchanged and compatible

## 1.4.0a0

Campaign Manifests + Resume milestone:

- schema-versioned campaign manifests
- canonical SHA-256 manifest fingerprints
- manifest tamper detection
- manifest-driven campaign execution
- manifest export from saved campaigns
- deterministic candidate-order randomization
- required seed for randomized campaigns
- per-replicate candidate-order audit trail
- resumable incomplete/error replicates
- preservation of completed optimization IDs
- automatic statistical-study rebuild after resume
- static hardware/software provenance compatibility guard
- explicit provenance-change override with resume audit trail
- `agoctl campaign manifest`
- `agoctl campaign resume`
- `agoctl campaign run --manifest`
- backward-compatible loading of v1.3 campaign JSON
- deterministic manifest, resume, randomization and CLI tests

## 1.3.0a0

Experiment Campaigns + Hardware/Software Provenance milestone:

- repeated optimizer campaigns
- campaign specification model
- per-replicate status and optimization IDs
- automatic one-group statistical study
- campaign-to-campaign study comparison
- GPU inventory provenance
- driver and CUDA provenance
- runtime, package-version and git provenance
- campaign benchmark environment variables
- user-defined campaign tags
- persisted campaign reports under `.ago/campaigns`
- `agoctl campaign run|list|show|compare`
- deterministic campaign, provenance, persistence and CLI tests
- no scheduler mutation or background execution

## 1.2.0a0

Multi-node Kubernetes Planning milestone:

- JobSet renderer using `jobset.x-k8s.io/v1alpha2`
- one replicated worker Job per requested node
- explicit GPU-per-node requests and limits
- world-size / node-shape validation
- deterministic coordinator DNS
- downward-API JobSet node rank
- distributed master address and port environment
- optional Kueue LocalQueue label
- optional service account
- Kueue queue label support for single Kubernetes Jobs
- `agoctl plan --kubernetes-mode jobset`
- render-only safety boundary preserved
- deterministic renderer and CLI tests

## 1.1.0a0

Reproducible Evidence Bundles milestone:

- portable evidence bundle model
- embedded autopilot decision artifact
- embedded baseline and candidate optimization artifacts
- optional embedded study, correlation and time-series evidence
- canonical JSON serialization for integrity hashing
- per-artifact SHA-256 fingerprints
- bundle-level SHA-256 fingerprint
- offline bundle verification
- tamper detection with dedicated CLI exit code
- runtime provenance capture
- environment-variable allowlist to avoid arbitrary secret capture
- user-defined reproducibility tags
- persisted bundles under `.ago/bundles`
- `agoctl bundle create|list|show|verify`
- deterministic hashing, provenance, persistence and CLI tests
- no scheduler mutation or external side effects

## 1.0.0a0

Recommend-only Closed-Loop Autopilot milestone:

- promote / hold / reject decision model
- strict guarded-validation integration
- recommend-only deployment-policy enforcement
- baseline and candidate deployment-plan generation
- plan diff included in every decision
- optional statistical-study evidence gates
- optional correlation evidence gates
- optional per-GPU time-series evidence gates
- Pareto candidate requirement
- minimum replicate and config-stability requirements
- straggler, memory-pressure and thermal hold conditions
- explicit operator-controlled rollback plan
- persisted autopilot reports under `.ago/autopilot`
- `agoctl autopilot decide|list|show`
- deterministic engine, persistence and CLI tests
- no automatic Kubernetes or Slurm mutation

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
