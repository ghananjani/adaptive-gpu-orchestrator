# Communication Baselines + Runtime Provenance

v1.5 ties workload-level campaign performance to a measured distributed-communication baseline.

## Why a communication baseline

Two GPU systems can run the same model on the same number of accelerators and still show very different scaling because of:

- NVLink / NVSwitch topology
- PCIe topology
- InfiniBand or Ethernet behavior
- NCCL library/runtime differences
- rank placement
- node-to-node bandwidth

A workload throughput number is therefore more useful when it can be interpreted alongside a communication baseline captured on the same environment.

## NCCL baseline specification

A campaign can optionally contain:

```text
binary
begin
end
factor
gpus
operation
```

Example:

```json
{
  "binary": "/opt/nccl-tests/build/all_reduce_perf",
  "begin": "8M",
  "end": "1G",
  "factor": 2,
  "gpus": 8,
  "operation": "all_reduce"
}
```

This specification is part of `CampaignSpec`, so it is included in the campaign-manifest SHA-256.

## Running from the CLI

```bash
agoctl campaign run \
  --name h100-llama70b \
  --group h100 \
  --repeats 5 \
  --gpus 8 \
  --tp 1,2,4,8 \
  --nccl-baseline \
  --nccl-binary /opt/nccl-tests/build/all_reduce_perf \
  --nccl-begin 8M \
  --nccl-end 1G \
  --nccl-factor 2 \
  --nccl-gpus 8 \
  -- python benchmark.py
```

The NCCL baseline runs synchronously before optimizer replicates.

If the baseline command fails or produces no parseable samples, campaign creation fails rather than silently recording an empty communication baseline.

## Captured artifact

`NCCLCommunicationBaseline` stores:

- the baseline specification
- every parsed `NCCLResult`
- summary statistics
- SHA-256 fingerprint

Summary fields include:

```text
sample_count
busbw_mean_gbps
busbw_max_gbps
busbw_min_gbps
algbw_mean_gbps
algbw_max_gbps
largest_message_bytes
largest_message_busbw_gbps
```

The SHA-256 covers the specification and raw measured result points. Summary values are derived from those points.

Loading a stored baseline verifies its fingerprint.

## Campaign manifests

When an NCCL baseline is enabled, the baseline **configuration** is part of the campaign manifest.

The measured baseline results are environment evidence and remain in the campaign report, not in the declarative manifest.

Running the same manifest on another machine therefore:

1. verifies the manifest
2. captures that machine's provenance
3. runs the same NCCL baseline parameters
4. runs the same optimizer campaign
5. stores a new measured communication artifact

This supports direct H100-vs-H200 or cluster-vs-cluster comparison.

## Campaign comparison

If the baseline campaign and a candidate campaign both contain NCCL baselines, `campaign compare` appends a descriptive finding such as:

```text
NCCL mean bus bandwidth h200 vs h100: +20.00% (70.00 -> 84.00 GB/s)
```

This does not claim communication bandwidth caused the workload-performance difference. It records a relevant measured systems difference alongside the statistical workload comparison.

## Runtime provenance

v1.5 adds `runtime_versions` to campaign provenance.

The collector probes:

### CUDA toolkit

When `nvcc` is available:

```bash
nvcc --version
```

The release number is recorded as:

```text
cuda_toolkit
```

### NVIDIA Python runtime packages

When installed, the provenance collector records package versions for:

- `nvidia-nccl-cu12`
- `nvidia-nccl-cu13`
- `nvidia-cuda-runtime-cu12`
- `nvidia-cuda-runtime-cu13`

Normalized runtime keys include:

```text
nccl_package
nccl_package_name
cuda_runtime_package
cuda_runtime_package_name
```

The existing package-version inventory still retains the complete tracked package names.

## Resume compatibility

The v1.4 resume guard compared hardware, driver, CUDA compatibility and tracked ML packages.

v1.5 also compares `runtime_versions`.

A campaign will therefore refuse an ordinary resume when the recorded CUDA toolkit or tracked NCCL/CUDA runtime identity changes.

The existing explicit override remains available:

```bash
agoctl campaign resume <campaign-id> --allow-provenance-change
```

The changed provenance remains recorded in the resume audit trail.

## Backward compatibility

Older campaign reports remain loadable.

Defaults:

```text
CampaignSpec.nccl_baseline = null
CampaignReport.communication_baseline = null
CampaignProvenance.runtime_versions = {}
```

## Correlation compatibility

The existing telemetry-correlation API still exposes `summarize_nccl`.

v1.5 moves the implementation to the benchmarks package so campaign baselines and correlation reports use identical summary logic.

## Safety

NCCL baseline execution is active benchmarking and can consume the GPUs selected by `--nccl-gpus`.

It is only run when explicitly requested by the campaign specification.

The feature does not:

- submit Kubernetes jobs
- submit Slurm jobs
- change GPU clocks
- change power caps
- change MIG configuration
- alter NCCL configuration
- run in the background

## Research workflow

A useful cross-hardware experiment is now:

```text
same version-controlled campaign manifest
       │
       ├── H100 system
       │     ├── hardware/runtime provenance
       │     ├── NCCL baseline
       │     └── repeated workload optimization
       │
       └── H200 system
             ├── hardware/runtime provenance
             ├── NCCL baseline
             └── repeated workload optimization

                    ↓
             campaign compare
                    ↓
 workload statistics + config stability + NCCL delta
```
