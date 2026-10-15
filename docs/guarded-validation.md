# Guarded Validation

v0.5 adds a promotion gate between optimization and any future scheduler mutation.

## Validation inputs

The CLI compares the `best` trial from two persisted optimization runs:

- baseline
- candidate

Each trial may contain throughput, p99 latency, energy, GPU utilization, memory pressure and exit status.

## Policy

`ValidationPolicy` can require:

- minimum throughput gain
- maximum p99-latency regression
- maximum energy regression
- maximum memory-pressure regression

Thresholds are percentages relative to the baseline.

Configured guardrails are strict. If a required metric is missing or the baseline value cannot be used to calculate a delta, the candidate is rejected rather than silently accepted.

## Decision

A validation report records:

- accept or reject
- baseline and candidate configurations
- per-metric percentage deltas
- rejection reasons
- rollback recommendation
- policy used for the decision

Rejected candidates recommend rollback by default. In v0.5 this is advisory only; the project still does not mutate scheduler state.

## Audit trail

Reports are stored as JSON under:

```text
.ago/validations/
```

They can be inspected with:

```bash
agoctl validations list
agoctl validations show <validation-id>
```

## Deployment plan diff

The Python API exposes `diff_plans()` to compare two `DeploymentPlan` objects. It reports changes to:

- backend
- parallelism configuration
- world size
- command
- GPU availability
- rank-to-GPU placement
- node selectors
- runtime environment

This provides the foundation for a future pre-deployment dry-run diff and an auditable rollback plan.

## Safety progression

Automatic scheduler mutation should remain disabled until a future controller can prove:

1. candidate metrics pass validation
2. the deployment diff is within policy
3. ownership boundaries are explicit
4. the prior plan is recoverable
5. rollback can be executed and verified
6. before/after metrics are recorded
