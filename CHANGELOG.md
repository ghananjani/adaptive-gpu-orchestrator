# Changelog

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
