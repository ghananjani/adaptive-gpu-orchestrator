# Cluster Controller

v0.4 introduces a scheduler-facing planning layer while deliberately keeping all execution **recommend-only**.

## Deployment plan

A plan contains:

- selected `ParallelismConfig`
- scheduler backend
- workload command
- declared GPU budget
- logical rank-to-GPU mapping
- AGO runtime environment
- optional node selectors
- safety warnings
- recommend-only state

## Policy

`DeploymentPolicy` currently supports:

- maximum GPU count
- exact-allocation requirement
- recommend-only mode

The planner rejects configurations that exceed either physical availability or policy limits.

## Topology-aware placement

If both a communication graph and physical topology are supplied through the Python API, the controller reuses the existing rank-placement optimizer.

Without both inputs, it uses deterministic sequential placement and records a warning in the plan.

## Kubernetes renderer

The current renderer emits a `batch/v1 Job` with:

- image
- workload command
- AGO environment
- GPU requests/limits using `nvidia.com/gpu`
- optional node selector
- plan ID labels

This v0.4 renderer is intentionally single-pod. Multi-node Kubernetes training will require a later adapter such as JobSet, Kubeflow Training Operator, or another distributed-workload controller.

## Slurm renderer

The renderer emits:

- shebang
- all `#SBATCH` directives before shell commands
- nodes
- GPUs per node
- task count
- optional partition
- AGO environment exports
- final `srun` command

The generated script is reviewable and reproducible but is not submitted automatically.

## Safety progression

A future mutating controller should only be enabled after the project has:

1. explicit ownership boundaries
2. dry-run diff output
3. policy-enforced resource ceilings
4. canary validation
5. rollback support
6. auditable before/after metrics

Until then, plan generation remains separated from scheduler execution.
