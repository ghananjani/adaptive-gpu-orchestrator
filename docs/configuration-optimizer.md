# Configuration Optimizer

The v0.3 optimizer turns Adaptive GPU Orchestrator from an observation tool into an experimentation engine.

## Model

Each candidate is represented as:

```text
tensor_parallel
pipeline_parallel
data_parallel
batch_size
concurrency
```

The world size is:

```text
TP × PP × DP
```

Candidates exceeding the declared GPU budget are discarded. An optional exact-world-size mode keeps only configurations that consume the entire GPU allocation.

## Canary contract

`CommandCanaryRunner` executes the same benchmark command once per candidate and injects the candidate through environment variables.

The command may be a wrapper around:

- vLLM
- TensorRT-LLM
- PyTorch Distributed
- DeepSpeed
- Megatron-LM
- an inference load generator
- a training-step microbenchmark

The command must print a JSON object containing measured metrics. Supported fields are:

```text
throughput_per_s
p99_latency_ms
energy_joules
gpu_util_pct
memory_pressure_pct
```

Only metrics required by the selected objective or constraints need to be present.

## Constraints

A configured constraint is strict. If its corresponding metric is missing, the trial is rejected rather than silently treated as valid.

Supported constraints:

- maximum p99 latency
- maximum energy
- minimum throughput
- maximum memory pressure

Non-zero benchmark exits and timeouts are also infeasible trials.

## Scoring

The optimizer normalizes feasible trials relative to the best observed value in the same run.

Objectives:

- **throughput** — maximize measured throughput
- **latency** — minimize p99 latency
- **energy** — minimize energy
- **balanced** — 50% throughput, 30% latency, 20% energy when those metrics exist

Balanced scoring renormalizes weights when a metric is unavailable.

## Research extension points

The current search is deliberately deterministic and transparent. Future work can replace candidate enumeration with:

- successive halving
- Bayesian optimization
- topology-aware pruning
- learned performance models
- online contextual bandits

The executor and scoring interfaces are separated so those research algorithms can evolve without coupling to Kubernetes, Slurm, vLLM or a particular accelerator generation.
