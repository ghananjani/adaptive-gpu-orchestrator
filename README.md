# Adaptive GPU Orchestrator for AI Workloads

**An autopilot for GPU clusters.** `adaptive_gpu_orchestrator` discovers GPU topology, captures live utilization, ingests communication benchmarks, diagnoses common bottlenecks, and lays the foundation for closed-loop optimization of distributed AI workloads.

> Status: **v0.1 pre-alpha**. The first milestone focuses on trustworthy observation and diagnosis before automatic remediation.

## Why this exists

Modern AI clusters already have schedulers, profilers, metrics stacks, and serving engines. What is still missing is a lightweight layer that connects them into a closed loop:

**observe → diagnose → experiment → optimize → schedule → validate → learn**

`adaptive_gpu_orchestrator` starts with the first two steps and is intentionally designed to grow into topology-aware placement, TP/PP/DP autotuning, KV-cache-aware scheduling, energy-aware orchestration, and learned performance prediction.

## v0.1 capabilities

- NVIDIA GPU discovery through `nvidia-smi`
- H100/H200/A100/etc. model, memory, utilization, temperature, power, PCI bus and compute-capability capture
- `nvidia-smi topo -m` parsing into GPU-to-GPU links
- NCCL `all_reduce_perf` wrapper + parser
- Rule-based diagnoses for:
  - GPU underutilization
  - memory pressure
  - communication-bound workloads
  - load imbalance / stragglers
  - thermal or power ceilings
- Human-readable and JSON CLI output
- Safe no-GPU mode for local development and CI
- Dependency-light Python API that works without CUDA libraries installed

## Quick start

```bash
agoctl inspect
agoctl topology
agoctl doctor
```

Machine-readable output:

```bash
agoctl --json inspect
agoctl --json doctor
```

Install locally:

```bash
pip install -e .
agoctl inspect
```

## Example output

```text
Host: gpu-node-01
Driver: 570.xx  CUDA: 12.x
GPUs: 8
  GPU 0: NVIDIA H100 80GB HBM3 | util=47.0% | memory=68231/81559 MiB
  ...

Bottleneck: communication_bound
Severity: medium
Low GPU utilization coincides with weak collective bandwidth.

Recommendations:
  - [high] Validate NCCL topology and rank placement
  - [medium] Profile collective time
```

## Python API

```python
from adaptive_gpu_orchestrator import Doctor

snapshot = Doctor().inspect()
print(snapshot.to_dict())

report = Doctor().diagnose()
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

Future releases will persist benchmark baselines per GPU model/topology and compare live measurements against hardware-specific expectations instead of generic warning thresholds.

## Design principles

1. **Measure first.** Automatic remediation without trustworthy observability is dangerous.
2. **Topology is a first-class input.** A GPU is not interchangeable with every other GPU in a distributed job.
3. **Explain every recommendation.** The controller should surface evidence, not output opaque advice.
4. **Degrade gracefully.** Developers should be able to run tests and reason about the code without access to an H100 cluster.
5. **Automate experiments, not assumptions.** Later optimizers will validate candidate configurations with canary runs.
6. **Stay scheduler-agnostic.** Kubernetes/Kueue, Slurm, Ray, and bare-metal adapters belong behind clean interfaces.

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

The platform is deliberately shaped around publishable systems questions:

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
Discovery, topology, utilization, NCCL parsing, explainable diagnosis.

### v0.2 — Workload Profiler
PyTorch/vLLM instrumentation, per-phase timing, DCGM telemetry, experiment persistence.

### v0.3 — Configuration Optimizer
Automated canaries over batch size, placement, TP/PP/DP and runtime settings.

### v0.4 — Cluster Controller
Kubernetes/Kueue + Slurm adapters, topology-aware placement, policy engine.

### v1.0 — Closed-loop Autopilot
Continuous observe/diagnose/experiment/optimize cycle with safety guardrails.

## Contributing

This project aims to be useful to both ML systems researchers and production GPU infrastructure teams. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT License.
