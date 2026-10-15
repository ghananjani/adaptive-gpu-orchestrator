# Adaptive GPU Orchestrator for AI Workloads

**An autopilot for GPU clusters.** `adaptive_gpu_orchestrator` discovers GPU topology, profiles distributed AI workloads, diagnoses bottlenecks, runs controlled canary experiments, and selects better GPU configurations.

> Status: **v0.3 pre-alpha — Configuration Optimizer.**

## Control loop

**observe → diagnose → experiment → optimize → schedule → validate → learn**

The project deliberately separates measurement, experimentation, and remediation so optimization decisions remain reproducible and explainable.

## Current capabilities

### GPU Doctor
- NVIDIA GPU discovery through `nvidia-smi`
- H100/H200/A100-class model, memory, utilization, temperature, power and PCI data
- NVLink/NVSwitch/PCIe topology parsing
- NCCL benchmark parsing and execution
- explainable detection of underutilization, memory pressure, communication stalls, imbalance and thermal/power ceilings
- topology-aware logical-rank placement baseline

### Workload Intelligence
- continuous telemetry around arbitrary commands
- `nvidia-smi` and optional NVML telemetry providers
- persisted workload profiles under `.ago/experiments`
- workload-level diagnosis
- reusable Python profiler API
- no-GPU CI/development mode

### Configuration Optimizer
- search spaces over tensor/pipeline/data parallelism
- batch-size and concurrency sweeps
- GPU-budget and exact-world-size validation
- command-based canary benchmarks
- SLO constraints for p99 latency, energy, throughput and memory pressure
- throughput, latency, energy and balanced objectives
- persisted optimization results under `.ago/optimizations`
- dry-run search inspection before expensive benchmarking

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

Profile a workload:

```bash
agoctl profile \
  --name llama70b-serving \
  --interval 0.5 \
  --tag model=llama70b \
  -- python serve.py
```

## Optimize a workload

First inspect the candidate search space without running anything:

```bash
agoctl optimize \
  --gpus 8 \
  --tp 1,2,4,8 \
  --pp 1,2 \
  --dp 1 \
  --batch 1,2,4,8 \
  --concurrency 1,8 \
  --dry-run
```

Then run canary benchmarks:

```bash
agoctl optimize \
  --gpus 8 \
  --tp 1,2,4,8 \
  --pp 1,2 \
  --batch 1,2,4,8 \
  --concurrency 1,8,32 \
  --objective balanced \
  --max-p99-ms 500 \
  --max-memory-pressure 92 \
  -- python benchmark.py
```

The benchmark receives:

```text
AGO_TP
AGO_TENSOR_PARALLEL
AGO_PP
AGO_PIPELINE_PARALLEL
AGO_DP
AGO_DATA_PARALLEL
AGO_BATCH_SIZE
AGO_CONCURRENCY
AGO_WORLD_SIZE
AGO_CONFIG_LABEL
```

It should emit one JSON metrics object on stdout:

```json
{
  "throughput_per_s": 14320,
  "p99_latency_ms": 417,
  "energy_joules": 4820,
  "gpu_util_pct": 91.4,
  "memory_pressure_pct": 87.2
}
```

See [`examples/benchmark_contract.py`](examples/benchmark_contract.py) and
[`docs/configuration-optimizer.md`](docs/configuration-optimizer.md).

Inspect saved runs:

```bash
agoctl optimizations list
agoctl optimizations show <optimization-id>
```

## Python API

```python
from adaptive_gpu_orchestrator import (
    ConfigurationOptimizer,
    OptimizationConstraints,
    OptimizationObjective,
    ParallelismConfig,
)
from adaptive_gpu_orchestrator.optimizer import CommandCanaryRunner

candidates = [
    ParallelismConfig(tensor_parallel=2, batch_size=2),
    ParallelismConfig(tensor_parallel=4, batch_size=4),
]

optimizer = ConfigurationOptimizer(
    runner=CommandCanaryRunner(["python", "benchmark.py"]),
    objective=OptimizationObjective.BALANCED,
    constraints=OptimizationConstraints(max_p99_latency_ms=500),
)

result = optimizer.optimize(candidates, baseline=candidates[0])
print(result.best.to_dict() if result.best else "no feasible configuration")
```

## NCCL benchmark ingestion

```bash
agoctl benchmark-nccl \
  --binary /opt/nccl-tests/build/all_reduce_perf \
  --begin 8M \
  --end 1G \
  --gpus 8
```

## Design principles

1. **Measure first.**
2. **Topology is a first-class input.**
3. **Explain every recommendation.**
4. **Degrade gracefully without GPU hardware.**
5. **Automate experiments, not assumptions.**
6. **Keep scheduler integrations behind clean interfaces.**

## Research directions

- topology-aware placement for distributed LLM workloads
- adaptive TP/PP/DP selection
- heterogeneous H100/H200 scheduling
- KV-cache-aware resource allocation
- SLO-aware energy optimization
- communication/computation overlap prediction
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
Kubernetes/Kueue + Slurm adapters and topology-aware policy execution.

### v1.0 — Closed-loop Autopilot
Continuous observe/diagnose/experiment/optimize with guarded remediation.

## License

MIT License.
