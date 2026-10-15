# Architecture

## Goal

`adaptive_gpu_orchestrator` is a control plane for understanding and eventually optimizing GPU workloads. It is not intended to replace Kubernetes, Slurm, Ray, vLLM, NCCL, DCGM, or PyTorch Profiler. It connects signals from those systems and turns them into explainable decisions.

## Control loop

1. **Discover** hardware, topology and runtime environment.
2. **Observe** workload and system telemetry.
3. **Diagnose** the dominant bottleneck with evidence.
4. **Generate** candidate configurations.
5. **Experiment** using short canary runs.
6. **Select** the best candidate for the requested objective.
7. **Deploy** through a scheduler adapter.
8. **Validate** that production metrics match the experiment.
9. **Learn** from historical runs.

v0.1 implements steps 1–3.

## Core domains

### Discovery

Produces a normalized graph of:

- GPU model / UUID / PCI identity
- memory capacity
- NVLink/NVSwitch relationships
- PCIe locality
- NUMA affinity (planned)
- InfiniBand/RDMA devices (planned)
- MIG partitions (planned)

### Telemetry

v0.1 consumes point-in-time values from `nvidia-smi`. v0.2 will add a sampling collector backed by NVML/DCGM.

Expected metrics include SM activity, tensor-core activity, HBM bandwidth, memory occupancy, power, temperature, PCIe, NVLink, ECC/XID, per-process attribution and workload-phase metadata.

### Diagnosis

Diagnosis must remain explainable. Every classification includes:

- a named bottleneck
- severity
- evidence signals
- recommended next measurements or actions

Rules are the first implementation. Later releases can layer statistical models on top, but opaque models should not be the only source of truth.

### Experiment engine (planned)

A candidate configuration is immutable and reproducible:

```text
hardware + model + workload shape + TP + PP + DP + batch + placement + runtime flags
```

The engine executes short canaries, records throughput/latency/energy/communication metrics, and compares candidates subject to workload constraints.

### Optimizer (planned)

The optimizer will support objective functions such as:

- maximize throughput
- minimize p99 latency
- minimize joules/token subject to an SLO
- minimize cost subject to throughput

Search begins with bounded heuristics/Bayesian optimization before learned performance prediction is introduced.

### Scheduler adapters (planned)

The core optimizer should not care whether a workload is launched through Kubernetes/Kueue, Slurm or Ray. Adapters translate a selected plan into scheduler-specific resources and placement constraints.

## Safety model

Automatic mutation is disabled until the controller can:

- produce a before/after experiment
- explain why a change is proposed
- limit changes to a declared policy envelope
- rollback automatically when validation fails
- avoid touching workloads outside its ownership boundary

The first cluster-controller releases should default to **recommend-only** mode.
