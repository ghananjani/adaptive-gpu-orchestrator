# Adaptive GPU Orchestrator for AI Workloads

**An autopilot for GPU clusters.** `adaptive_gpu_orchestrator` discovers GPU topology, profiles workloads, diagnoses bottlenecks, runs canary optimization, and renders safe deployment plans for GPU schedulers.

> Status: **v0.5 pre-alpha — Guarded Validation.**

## Control loop

**observe → diagnose → experiment → optimize → plan → validate → learn**

The controller remains **recommend-only** in v0.4: it generates deployable artifacts but does not mutate Kubernetes or Slurm state.

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

Optimize:

```bash
agoctl optimize \
  --gpus 8 \
  --tp 1,2,4,8 \
  --pp 1,2 \
  --batch 1,2,4,8 \
  --objective balanced \
  -- python benchmark.py
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

The output is a ready-to-review Slurm script. It is **not submitted automatically**.

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

## Plan from an optimizer winner

```bash
agoctl plan \
  --backend slurm \
  --available-gpus 8 \
  --optimization <optimization-id> \
  --nodes 1 \
  --gpus-per-node 8 \
  -- python benchmark_or_train.py
```


## Validate an optimizer candidate before promotion

Compare two saved optimizer winners:

```bash
agoctl validate \
  --baseline-optimization baseline-run-id \
  --candidate-optimization candidate-run-id \
  --min-throughput-gain 5 \
  --max-p99-regression 5 \
  --max-energy-regression 5
```

A rejected candidate returns a nonzero exit code and records why promotion was blocked. By default, rejection also marks rollback as recommended for future mutating controller integrations.

Inspect validation history:

```bash
agoctl validations list
agoctl validations show <validation-id>
```

See [`docs/guarded-validation.md`](docs/guarded-validation.md).

## Safety model

v0.4 intentionally stops before scheduler mutation:

- no `kubectl apply`
- no Slurm submission
- no automatic eviction or migration
- no changes outside the generated plan
- policy checks run before rendering
- recommend-only state is included in the plan

See [`docs/cluster-controller.md`](docs/cluster-controller.md).

## Python API

```python
from adaptive_gpu_orchestrator import (
    DeploymentPlanner,
    DeploymentPolicy,
    ParallelismConfig,
    SchedulerBackend,
)

plan = DeploymentPlanner(
    DeploymentPolicy(max_gpus=8, recommend_only=True)
).plan(
    config=ParallelismConfig(tensor_parallel=4, pipeline_parallel=2),
    backend=SchedulerBackend.SLURM,
    command=["python", "train.py"],
    available_gpus=8,
)

print(plan.to_dict())
```

## Research directions

- topology-aware placement for distributed LLM workloads
- adaptive TP/PP/DP selection
- heterogeneous H100/H200 scheduling
- KV-cache-aware resource allocation
- SLO-aware energy optimization
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

### v1.0 — Closed-loop Autopilot
Guarded observe/diagnose/experiment/optimize/deploy/validate control loop.

## License

MIT License.
