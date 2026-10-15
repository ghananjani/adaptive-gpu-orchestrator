# Security

`adaptive_gpu_orchestrator` interacts with low-level system tools and will eventually integrate with cluster schedulers. Treat its output as operational guidance, not authorization to mutate production infrastructure.

## v0.1 guarantees

- discovery and diagnosis are read-only
- the NCCL wrapper only executes the explicitly configured benchmark binary
- no scheduler or cluster state is modified
- no telemetry leaves the host

Do not include credentials, proprietary model data, internal hostnames, or sensitive production traces in public bug reports.
