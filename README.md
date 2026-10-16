# Adaptive GPU Orchestrator for AI Workloads

**An autopilot for GPU clusters.** `adaptive_gpu_orchestrator` discovers GPU topology, profiles workloads, diagnoses bottlenecks, normalizes native workload metrics, runs canary optimization, and renders safe deployment plans for GPU schedulers.

> Status: **v1.3 alpha — Experiment Campaigns + Hardware/Software Provenance.**

## Control loop

**observe → diagnose → experiment → optimize → plan → validate → learn**

The controller remains recommend-only: it generates deployment artifacts and promotion decisions but does not mutate Kubernetes or Slurm state.

## Current capabilities

### GPU Doctor
- NVIDIA GPU discovery through `nvidia-smi`
- H100/H200/A100-class utilization, memory, thermal and power data
- NVLink/NVSwitch/PCIe topology parsing
- NCCL benchmark parsing and execution
- explainable bottleneck diagnosis
- topology-aware rank-placement baseline

### Workload Intelligence
- continuous telemetry around arbitrary commands
- `nvidia-smi` and optional NVML providers
- persisted workload profiles
- workload-level diagnosis

### Workload Adapters
- generic JSON benchmark contract
- vLLM serving metrics normalization
- PyTorch-distributed training metrics normalization
- preservation of adapter-native metrics in `TrialMetrics.extras`
- TTFT / TPOT / ITL retention for LLM serving
- communication / compute / phase timing retention for distributed training
- `agoctl optimize --adapter ...`

### Telemetry Correlation
- correlate a saved optimizer winner with a saved workload profile
- optional NCCL baseline ingestion
- derived energy estimates from power telemetry
- GPU utilization, imbalance, memory-pressure and thermal signals
- adapter-native metric carry-through
- descriptive, non-causal findings
- persisted reports under `.ago/correlations`
- `agoctl correlate` and `agoctl correlations list/show`

### DCGM + Time-Series Analysis
- per-GPU telemetry retained in every new workload sample
- backward-compatible loading of aggregate-only historical profiles
- DCGM Exporter/Prometheus ingestion
- `dcgmi dmon` trace ingestion
- persistent GPU-utilization straggler detection
- memory-pressure and thermal time-series findings
- user-defined phase windows aligned by profile offset
- optional DCGM cross-checks
- persisted reports under `.ago/timeseries`
- `agoctl timeseries analyze|list|show`

### Statistical Experiment Studies
- repeated-run aggregation over saved optimization results
- Student-t 95% confidence intervals
- baseline-vs-candidate percent changes
- direction-aware relative improvement
- Cohen's d effect size
- selected-configuration stability
- throughput/latency/energy Pareto frontier
- adapter-specific metric aggregation
- persisted reports under `.ago/studies`
- `agoctl study analyze` and `agoctl studies list/show`

### Closed-Loop Autopilot
- one auditable decision over optimization, validation, studies, telemetry and scheduler policy
- explicit **promote / hold / reject** outcomes
- baseline-vs-candidate deployment-plan diff
- optional statistical-study, correlation and per-GPU time-series evidence gates
- Pareto-frontier and config-stability gating
- thermal, memory-pressure and straggler holds
- operator-controlled rollback plan to the saved baseline configuration
- persisted reports under `.ago/autopilot`
- `agoctl autopilot decide|list|show`
- hard safety boundary: no Kubernetes or Slurm mutation from the autopilot process

### Reproducible Evidence Bundles
- portable snapshot of an autopilot decision and every referenced evidence artifact
- embedded baseline and candidate optimization results
- optional embedded statistical study, correlation and time-series reports
- canonical JSON SHA-256 fingerprint per artifact
- bundle-level SHA-256 fingerprint
- offline integrity verification without the original `.ago` stores
- runtime provenance: Python, platform, package version and git commit when available
- safe environment whitelist rather than arbitrary environment capture
- user-defined reproducibility tags
- persisted bundles under `.ago/bundles`
- `agoctl bundle create|list|show|verify`

### Multi-node Kubernetes Planning
- single-pod `batch/v1 Job` remains the default Kubernetes mode
- multi-node `jobset.x-k8s.io/v1alpha2 JobSet` rendering
- one Job per node with explicit GPU-per-node resource requests/limits
- strict `nodes × GPUs-per-node = world size` validation
- deterministic coordinator DNS derived from the JobSet name
- downward-API node-rank exposure from JobSet job-index labels
- distributed `MASTER_ADDR` / `MASTER_PORT` environment
- optional Kueue LocalQueue label
- optional service account
- node selectors carried into every worker pod
- render-only safety boundary: no `kubectl apply`

### Experiment Campaigns
- repeated optimizer runs under one campaign identity
- campaign-level search specification, adapter, objective, constraints and command
- per-replicate optimization persistence
- automatic one-group statistical study for completed replicates
- direct campaign-to-campaign statistical comparison
- GPU inventory, driver and CUDA provenance from the existing discovery layer
- Python/runtime/git provenance
- installed ML-system package versions when available
- user-defined tags propagated into campaign metadata and benchmark environment
- persisted reports under `.ago/campaigns`
- `agoctl campaign run|list|show|compare`

### Configuration Optimizer
- TP/PP/DP + batch/concurrency search spaces
- canary benchmarks
- throughput, latency, energy and balanced objectives
- SLO constraints
- persisted optimization history
- `agoctl optimize`

### Cluster Controller
- deployment policy validation
- recommend-only plans
- optional reuse of topology-aware rank placement
- Kubernetes Job rendering
- Slurm script rendering
- plans generated from either direct configs or saved optimizer winners
- node selectors, Slurm partitions, node counts and GPU-per-node controls

### Guarded Validation
- baseline-vs-candidate promotion checks
- configurable throughput-gain and regression budgets
- strict metric requirements when a guardrail is configured
- explicit accept/reject decision
- rollback recommendation on rejected candidates
- persisted validation audit trail under `.ago/validations`
- deployment-plan diff utility
- `agoctl validate` and `agoctl validations list/show`

## Quick start

```bash
pip install -e .
```

Inspect and diagnose:

```bash
agoctl inspect
agoctl topology
agoctl doctor
```

Profile:

```bash
agoctl profile --name llama70b -- python serve.py
```

## Run a reproducible experiment campaign

A campaign repeats the existing optimizer search while preserving the exact search space, command, environment tags and hardware/software provenance:

```bash
agoctl campaign run \
  --name h100-llama70b \
  --group h100 \
  --repeats 5 \
  --gpus 8 \
  --tp 1,2,4,8 \
  --pp 1,2 \
  --batch 1,2,4,8 \
  --adapter vllm \
  --objective balanced \
  --max-p99-ms 500 \
  --tag hardware=h100 \
  --tag model=llama70b \
  -- python benchmark_vllm.py
```

Every replicate is stored as a normal optimization result. The campaign also records GPU model/count, driver, CUDA version, runtime/git metadata and installed package versions such as PyTorch, vLLM and Triton when present.

The benchmark child process receives:

```text
AGO_CAMPAIGN_ID
AGO_CAMPAIGN_NAME
AGO_CAMPAIGN_GROUP
AGO_REPLICATE_INDEX
AGO_CAMPAIGN_TAG_<TAG>
```

Compare two completed campaigns directly:

```bash
agoctl campaign compare <h100-campaign-id> <h200-campaign-id>
```

The comparison reuses the existing statistical-study engine, so the first campaign is the baseline group and later campaigns are compared with confidence intervals, direction-aware improvements, effect sizes, config-selection stability and Pareto analysis.

Inspect history:

```bash
agoctl campaign list
agoctl campaign show <campaign-id>
```

See [`docs/experiment-campaigns.md`](docs/experiment-campaigns.md).

## Optimize native vLLM benchmark output

Your benchmark command should print its final result as a JSON object on stdout.

```bash
agoctl optimize \
  --gpus 8 \
  --tp 1,2,4,8 \
  --pp 1,2 \
  --batch 1,2,4,8 \
  --concurrency 1,8,32 \
  --adapter vllm \
  --objective balanced \
  --max-p99-ms 500 \
  -- python benchmark_vllm.py
```

For vLLM, optimizer throughput prefers total-token throughput, then output-token throughput, then request throughput. Optimizer p99 latency prefers end-to-end p99 when available and otherwise uses p99 TTFT. Native TTFT, TPOT and ITL values remain in `TrialMetrics.extras`.

## Optimize distributed PyTorch training

```bash
agoctl optimize \
  --gpus 8 \
  --tp 1,2,4,8 \
  --dp 1,2,4 \
  --adapter pytorch-distributed \
  --objective throughput \
  -- python distributed_benchmark.py
```

The PyTorch adapter accepts common summary keys such as `tokens_per_s`, `samples_per_s`, `p99_step_time_ms`, `communication_time_ms`, `compute_time_ms`, energy, GPU utilization and memory pressure. Phase timing fields are preserved for later analysis.

See [`docs/workload-adapters.md`](docs/workload-adapters.md).

## Generic optimizer contract

The original generic JSON contract still works unchanged:

```bash
agoctl optimize \
  --gpus 8 \
  --tp 1,2,4,8 \
  --adapter generic \
  -- python benchmark.py
```

Expected normalized fields include:

```json
{
  "throughput_per_s": 14320,
  "p99_latency_ms": 417,
  "energy_joules": 4820,
  "gpu_util_pct": 91.4,
  "memory_pressure_pct": 87.2
}
```

## Render a Slurm plan

```bash
agoctl plan \
  --backend slurm \
  --available-gpus 8 \
  --tp 4 \
  --pp 2 \
  --nodes 2 \
  --gpus-per-node 4 \
  --partition gpu \
  -- python train.py
```

The output is a ready-to-review Slurm script. It is not submitted automatically.

## Render a multi-node Kubernetes JobSet

Use the same optimizer-selected configuration while splitting the world size across multiple nodes:

```bash
agoctl plan \
  --backend kubernetes \
  --available-gpus 8 \
  --tp 8 \
  --image myregistry/train:latest \
  --kubernetes-mode jobset \
  --nodes 2 \
  --gpus-per-node 4 \
  --queue-name gpu-queue \
  --node-selector accelerator=h100 \
  -- python train.py
```

The renderer emits a `jobset.x-k8s.io/v1alpha2` JobSet with two worker Jobs, each requesting four GPUs. It also exposes:

- `AGO_NODE_COUNT`
- `AGO_GPUS_PER_NODE`
- `AGO_NODE_RANK` from the JobSet job-index label
- `AGO_MASTER_ADDR`
- `MASTER_ADDR`
- `MASTER_PORT`

If `--gpus-per-node` is omitted, the CLI derives it when the configuration world size is evenly divisible by `--nodes`.

Add `--queue-name <local-queue>` to place the JobSet in a Kueue LocalQueue, and `--service-account-name` when the worker pods need an explicit service account.

See [`docs/multinode-kubernetes.md`](docs/multinode-kubernetes.md).

## Render a Kubernetes plan

```bash
agoctl plan \
  --backend kubernetes \
  --available-gpus 8 \
  --tp 8 \
  --image myregistry/llm:latest \
  --node-selector accelerator=h100 \
  -- python serve.py
```

The current Kubernetes renderer emits a single-pod `batch/v1 Job`; the target node therefore needs enough GPUs for the requested world size.

## Create a reproducible evidence bundle

After an autopilot decision is saved, package the decision and all of its referenced evidence into one portable JSON artifact:

```bash
agoctl bundle create \
  --autopilot <decision-id> \
  --tag hardware=h100 \
  --tag workload=llama70b
```

The bundle automatically loads the baseline and candidate optimization runs referenced by the decision and, when present, its study, correlation and time-series evidence. Each embedded artifact gets a canonical SHA-256 fingerprint, and the complete bundle gets its own SHA-256 fingerprint.

Verify a saved bundle without needing the original evidence stores:

```bash
agoctl bundle verify <bundle-id>
```

Inspect or list bundles:

```bash
agoctl bundle show <bundle-id>
agoctl bundle list
```

Integrity verification returns exit code `0` when valid and `5` when either an embedded artifact fingerprint or the bundle fingerprint does not match.

Provenance captures Python/runtime information, package version, git commit when available, user-provided tags, and only a small allowlist of environment variables such as `CUDA_VISIBLE_DEVICES` and selected NCCL settings. Arbitrary environment variables and secrets are not collected.

See [`docs/evidence-bundles.md`](docs/evidence-bundles.md).

## Make a closed-loop autopilot decision

The autopilot combines the evidence produced by the earlier milestones into one recommend-only decision:

```bash
agoctl autopilot decide \
  --baseline-optimization baseline-run \
  --candidate-optimization candidate-run \
  --backend slurm \
  --available-gpus 8 \
  --min-throughput-gain 5 \
  --max-p99-regression 5 \
  --study study-id \
  --require-study \
  --correlation correlation-id \
  --require-correlation \
  --timeseries timeseries-id \
  --require-timeseries \
  -- python train.py
```

Decision semantics:

- **promote** — guarded validation and every configured evidence gate passed.
- **hold** — validation passed, but evidence is insufficient or operational risk remains, such as low replicate count, unstable config selection, stragglers, high memory pressure, high temperature, mismatched evidence, or no meaningful plan change.
- **reject** — the candidate violates an explicit guarded-validation threshold.

Every report stores both scheduler plans, their diff, the validation report, evidence IDs, policy reasoning, and an operator-controlled rollback recipe that targets the baseline configuration.

Exit codes are `0` for promote, `4` for hold, and `3` for reject.

Inspect history:

```bash
agoctl autopilot list
agoctl autopilot show <decision-id>
```

See [`docs/closed-loop-autopilot.md`](docs/closed-loop-autopilot.md).

## Compare repeated experiment groups

Use saved optimization runs as replicates:

```bash
agoctl study analyze \
  --group h100=run-a1,run-a2,run-a3 \
  --group h200=run-b1,run-b2,run-b3
```

The first group is treated as the comparison baseline. Each group reports metric means, 95% confidence intervals when at least two replicates exist, selected-config stability, and adapter-specific metrics. Candidate groups are compared to the baseline with percent change, direction-aware relative improvement and Cohen's d when enough replicates exist.

When every run exposes throughput, p99 latency and energy, the study also computes a non-dominated Pareto frontier.

Inspect history:

```bash
agoctl studies list
agoctl studies show <study-id>
```

See [`docs/statistical-studies.md`](docs/statistical-studies.md).

## Analyze per-GPU time series

Profiles collected with v0.8 retain per-GPU telemetry while preserving the existing aggregate summary fields.

```bash
agoctl timeseries analyze \
  --profile <experiment-id> \
  --phase warmup:0:30 \
  --phase steady:30:120 \
  --straggler-gap 15
```

Add a DCGM Exporter capture:

```bash
agoctl timeseries analyze \
  --profile <experiment-id> \
  --dcgm-file dcgm.prom \
  --dcgm-format exporter \
  --dcgm-interval 1
```

Or parse `dcgmi dmon` output:

```bash
agoctl timeseries analyze \
  --profile <experiment-id> \
  --dcgm-file dmon.txt \
  --dcgm-format dmon
```

Inspect history:

```bash
agoctl timeseries list
agoctl timeseries show <report-id>
```

See [`docs/dcgm-timeseries.md`](docs/dcgm-timeseries.md).

## Correlate optimizer results with telemetry

Combine a saved optimizer winner with a saved workload profile:

```bash
agoctl correlate \
  --optimization <optimization-id> \
  --profile <experiment-id> \
  --nccl-output nccl-all-reduce.txt
```

The report combines normalized trial metrics, adapter-specific metrics, GPU utilization, utilization imbalance, memory pressure, temperature, power, an estimated energy value, and optional NCCL bandwidth statistics.

Correlation findings are intentionally descriptive rather than causal. They identify signals worth investigating; they do not claim that a specific telemetry signal caused the measured throughput or latency.

Inspect saved reports:

```bash
agoctl correlations list
agoctl correlations show <correlation-id>
```

See [`docs/telemetry-correlation.md`](docs/telemetry-correlation.md).

## Validate an optimizer candidate before promotion

```bash
agoctl validate \
  --baseline-optimization baseline-run-id \
  --candidate-optimization candidate-run-id \
  --min-throughput-gain 5 \
  --max-p99-regression 5 \
  --max-energy-regression 5
```

A rejected candidate returns a nonzero exit code and records why promotion was blocked.

## Safety model

The project intentionally stops before scheduler mutation:

- no `kubectl apply`
- no Slurm submission
- no automatic eviction or migration
- no changes outside generated artifacts
- policy checks run before rendering
- validation gates run before future promotion
- recommend-only state remains explicit

## Research directions

- topology-aware placement for distributed LLM workloads
- adaptive TP/PP/DP selection
- heterogeneous H100/H200 scheduling
- KV-cache-aware resource allocation
- TTFT / ITL / TPOT aware serving optimization
- communication/computation overlap prediction
- scheduler-aware canary validation
- learned performance models that reduce search cost

See [`docs/research-roadmap.md`](docs/research-roadmap.md).

## Roadmap

### v0.1 — GPU Doctor
Discovery, topology, NCCL parsing and explainable diagnosis.

### v0.2 — Workload Intelligence
Continuous telemetry, profiles, NVML support and experiment persistence.

### v0.3 — Configuration Optimizer
Canary search over parallelism, batch/concurrency and multi-objective constraints.

### v0.4 — Cluster Controller
Recommend-only Kubernetes/Slurm planning and deployment policy validation.

### v0.5 — Guarded Validation
Promotion gates, audit reports, rollback recommendations and plan diffs.

### v0.6 — Workload Adapters
Native vLLM and distributed PyTorch metric normalization with preserved research metrics.

### v0.7 — Telemetry Correlation
Offline correlation of optimizer results with GPU telemetry, adapter metrics and optional NCCL baselines.

### v0.8 — DCGM + Time-Series Analysis
Per-GPU telemetry, DCGM ingestion, phase alignment and persistent straggler analysis.

### v0.9 — Statistical Experiment Studies
Repeated-run confidence intervals, effect sizes, configuration stability and Pareto analysis.

### v1.0 — Closed-loop Autopilot
Recommend-only promote/hold/reject orchestration across optimization, statistical evidence, telemetry, scheduler planning, validation and rollback.

### v1.1 — Reproducible Evidence Bundles
Portable integrity-checked decision/evidence bundles with runtime provenance and offline verification.

### v1.2 — Multi-node Kubernetes Planning
Render-only JobSet/Kueue planning with explicit node/GPU topology and distributed-launcher environment.

### v1.3 — Experiment Campaigns + Provenance
Repeated optimization campaigns with hardware/software provenance and direct campaign-to-campaign statistical comparison.

## License

MIT License.
