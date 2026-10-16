# Experiment Campaigns + Hardware/Software Provenance

v1.3 adds a reproducible layer above the existing optimizer and statistical-study subsystems.

## Campaign model

A campaign stores the complete repeated experiment definition:

- campaign name
- statistical group label
- repeat count
- available GPU count
- candidate parallelism configurations
- optimization objective
- optimization constraints
- metrics adapter
- benchmark command
- timeout
- user-defined tags

The campaign itself does not introduce a second optimization algorithm. Every replicate calls the existing `ConfigurationOptimizer`.

## Repeated execution

Each replicate receives a new optimizer result ID and is persisted under:

```text
.ago/optimizations/
```

The campaign stores a per-replicate record with:

- replicate number
- status
- optimization ID
- selected configuration
- error text when execution failed

Replicate statuses are:

- `completed`
- `no_feasible_candidate`
- `error`

An error in one replicate does not erase successful prior replicates.

## Automatic statistical study

When at least one replicate produces a feasible winner, v1.3 automatically builds a one-group `StudyReport` from the completed optimization runs.

This immediately exposes:

- metric means
- Student-t 95% confidence intervals
- selected-config stability
- adapter-native metrics
- Pareto-compatible optimization data

The study is stored under:

```text
.ago/studies/
```

## Campaign comparison

`agoctl campaign compare` loads completed optimization IDs from two or more campaigns and passes them to the existing study engine.

Example:

```bash
agoctl campaign compare h100-id h200-id
```

The first campaign is the baseline study group.

Campaign group labels must be unique in one comparison.

This makes hardware comparison workflows explicit, for example:

```text
H100 campaign
  ├─ replicate 1 optimization
  ├─ replicate 2 optimization
  └─ replicate 3 optimization

H200 campaign
  ├─ replicate 1 optimization
  ├─ replicate 2 optimization
  └─ replicate 3 optimization

        ↓

Statistical Study
  throughput
  latency
  energy
  effect size
  config stability
  Pareto frontier
```

## Hardware provenance

Campaign creation reuses the existing `NvidiaSMI` discovery layer.

The stored cluster snapshot can include:

- hostname
- GPU count
- GPU model
- UUID
- total memory
- PCI bus ID
- compute capability
- utilization snapshot
- temperature/power snapshot
- topology
- NVIDIA driver version
- CUDA version
- discovery warnings

On hosts without `nvidia-smi`, the campaign still works and stores the no-GPU discovery warning.

## Software provenance

Campaign provenance also includes the existing runtime provenance:

- Python version
- Python implementation
- platform
- machine architecture
- installed adaptive-gpu-orchestrator version
- git commit when available
- whitelisted environment variables
- user tags

The campaign additionally records installed versions, when present, for:

- torch
- vllm
- triton
- nvidia-ml-py
- numpy

Missing optional packages are simply omitted.

## Benchmark environment

Each benchmark process receives the standard optimizer environment plus:

```text
AGO_CAMPAIGN_ID
AGO_CAMPAIGN_NAME
AGO_CAMPAIGN_GROUP
AGO_REPLICATE_INDEX
```

User tags are also exposed using normalized keys:

```text
AGO_CAMPAIGN_TAG_HARDWARE=h100
AGO_CAMPAIGN_TAG_MODEL=llama70b
```

This makes benchmark scripts able to stamp their own native logs without reading the campaign database.

## CLI

Run:

```bash
agoctl campaign run \
  --name h100-llama70b \
  --group h100 \
  --repeats 5 \
  --gpus 8 \
  --tp 1,2,4,8 \
  --adapter vllm \
  --tag hardware=h100 \
  -- python benchmark.py
```

List:

```bash
agoctl campaign list
```

Show:

```bash
agoctl campaign show <campaign-id>
```

Compare:

```bash
agoctl campaign compare <campaign-a> <campaign-b> [campaign-c ...]
```

## Storage

Campaign reports are stored under:

```text
.ago/campaigns/
```

They reference ordinary optimizer and study artifacts rather than embedding copies, keeping existing subsystem ownership intact.

## Safety boundary

Campaigns run benchmark commands synchronously because benchmarking is their explicit purpose.

They do not:

- submit Kubernetes workloads
- submit Slurm jobs
- apply JobSets
- mutate cluster schedulers
- run in the background
- automatically change GPU settings
- automatically set power caps, clocks or MIG configuration

Scheduler planning remains recommend-only.

## Research workflow

v1.3 supports a concrete H100-vs-H200 workflow:

1. run the same campaign specification on H100 hardware with `--group h100`
2. run it on H200 hardware with `--group h200`
3. preserve the automatically captured hardware/software provenance
4. compare the two campaign IDs
5. archive the resulting study or feed its optimizer runs into later autopilot/evidence workflows

This separates measured performance claims from environment ambiguity.

## Future extensions

- explicit campaign manifests checked into git
- randomized candidate execution order
- warmup/cooldown phases
- configurable inter-replicate delays
- automatic NCCL baseline capture per campaign
- container-image digest capture
- CUDA toolkit/NCCL library version probing
- campaign resume after interruption
- multi-host campaign federation
- CSV/Parquet export
