# Adaptive GPU Orchestrator for AI Workloads

**An autopilot for GPU clusters.** `adaptive_gpu_orchestrator` discovers GPU topology, continuously profiles GPU workloads, persists experiments, diagnoses bottlenecks, and provides the foundation for closed-loop optimization of distributed AI training and LLM inference.

> Status: **v0.2 pre-alpha — Workload Intelligence.**

## Why this exists

Modern AI clusters already have schedulers, profilers, metrics stacks, and serving engines. What is still missing is a lightweight layer that connects them into a closed loop:

**observe → diagnose → experiment → optimize → schedule → validate → learn**

Adaptive GPU Orchestrator is designed to become that layer while remaining scheduler-agnostic and explainable.

## Current capabilities

### GPU Doctor

- NVIDIA GPU discovery through `nvidia-smi`
- H100/H200/A100/etc. model, memory, utilization, temperature, power, PCI bus and compute-capability capture
- NVLink/NVSwitch/PCIe topology parsing
- NCCL `all_reduce_perf` parsing and benchmark execution
- explainable diagnosis of:
  - GPU underutilization
  - memory pressure
  - communication bottlenecks
  - load imbalance / stragglers
  - thermal and power ceilings
- topology-aware logical-rank placement optimization

### Workload Intelligence

- continuous telemetry sampling around an arbitrary workload
- dependency-free `nvidia-smi` telemetry provider
- optional lower-overhead NVML provider via `nvidia-ml-py`
- workload profiles with duration, command, tags and telemetry time series
- experiment persistence under `.ago/experiments`
- workload-level bottleneck diagnosis
- reusable Python profiling API
- no-GPU degradation for development and CI

## Quick start

Install locally:

```bash
pip install -e .
```

Inspect a GPU host:

```bash
agoctl inspect
agoctl topology
agoctl doctor
```

Profile an AI workload:

```bash
agoctl profile \
  --name llama70b-serving \
  --interval 0.5 \
  --tag model=llama70b \
  --tag environment=benchmark \
  -- python serve.py
```

Or collect a timed observation window without launching a command:

```bash
agoctl profile --name cluster-baseline --duration 30 --interval 1
```

Use NVML sampling:

```bash
pip install -e '.[nvml]'
agoctl profile --provider nvml --name training-run -- python train.py
```

Inspect saved experiments:

```bash
agoctl experiments list
agoctl experiments show <experiment-id>
```

Diagnose a previously captured workload:

```bash
agoctl doctor --profile <experiment-id>
```

Machine-readable output is available with the global `--json` flag:

```bash
agoctl --json inspect
agoctl --json experiments list
```

## Python API

```python
from adaptive_gpu_orchestrator import (
    Doctor,
    ExperimentStore,
    WorkloadProfiler,
)

profiler = WorkloadProfiler()
profile = profiler.run(
    name="llama-inference",
    command=["python", "serve.py"],
    duration_s=60,
    interval_s=1,
)

store = ExperimentStore()
store.save(profile)

report = Doctor().diagnose_profile(profile)
print(report.to_dict())
```

## NCCL benchmark ingestion

If `nccl-tests` is installed:

```bash
agoctl benchmark-nccl \
  --binary /opt/nccl-tests/build/all_reduce_perf \
  --begin 8M \
  --end 1G \
  --gpus 8
```

## Design principles

1. **Measure first.** Automatic remediation without trustworthy observability is dangerous.
2. **Topology is a first-class input.** A GPU is not interchangeable with every other GPU in a distributed job.
3. **Explain every recommendation.** The controller should surface evidence, not opaque advice.
4. **Degrade gracefully.** Developers can run tests and reason about the system without H100 access.
5. **Automate experiments, not assumptions.** Candidate optimizations should be validated by canary runs.
6. **Stay scheduler-agnostic.** Kubernetes/Kueue, Slurm, Ray and bare-metal adapters belong behind clean interfaces.

## Planned architecture

```text
                    ┌──────────────────────────────┐
                    │      Workload / Operator     │
                    └──────────────┬───────────────┘
                                   │
             ┌─────────────────────▼─────────────────────┐
             │           Adaptive Control Plane          │
             │  diagnose · experiment · optimize · learn │
             └──────────────┬──────────────┬─────────────┘
                            │              │
                   ┌────────▼──────┐ ┌────▼───────────┐
                   │   Telemetry   │ │  Experiment DB │
                   │ NVML/DCGM/etc │ │ + Baselines    │
                   └────────┬──────┘ └────┬───────────┘
                            │              │
             ┌──────────────▼──────────────▼─────────────┐
             │              GPU Cluster                  │
             │ NVLink · NVSwitch · PCIe · InfiniBand    │
             └───────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the deeper design.

## Research directions

- topology-aware placement for distributed LLM workloads
- automated TP/PP/DP configuration search
- heterogeneous H100/H200 scheduling
- KV-cache-pressure-aware resource allocation
- SLO-aware energy optimization
- communication/computation overlap prediction
- runtime detection of stragglers and degraded links
- learned performance models that avoid exhaustive search

See [`docs/research-roadmap.md`](docs/research-roadmap.md).

## Roadmap

### v0.1 — GPU Doctor
Discovery, topology, NCCL parsing, explainable diagnosis and placement baseline.

### v0.2 — Workload Intelligence
Continuous telemetry, workload profiles, optional NVML, experiment persistence and profile diagnosis.

### v0.3 — Configuration Optimizer
Automated canaries over batch size, placement, TP/PP/DP and runtime settings.

### v0.4 — Cluster Controller
Kubernetes/Kueue + Slurm adapters, topology-aware placement and policy engine.

### v1.0 — Closed-loop Autopilot
Continuous observe/diagnose/experiment/optimize cycle with safety guardrails.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT License.
