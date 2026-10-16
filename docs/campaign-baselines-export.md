# Campaign Baselines + Research Export

v1.5 adds campaign-level communication baselines and analysis-ready exports without changing the v1 campaign manifest schema.

## NCCL baseline

A campaign can attach one all-reduce baseline before its optimization replicates.

Run it directly:

```bash
agoctl campaign run \
  --name h100-study \
  --group h100 \
  --gpus 8 \
  --nccl-baseline \
  --nccl-binary /opt/nccl-tests/build/all_reduce_perf \
  --nccl-begin 8M \
  --nccl-end 1G \
  --nccl-factor 2 \
  --nccl-gpus 8 \
  -- python benchmark.py
```

Or ingest a previously captured file:

```bash
agoctl campaign run \
  --name h100-study \
  --group h100 \
  --gpus 8 \
  --nccl-output artifacts/allreduce.txt \
  -- python benchmark.py
```

The two source modes are mutually exclusive.

The stored baseline contains:

- source type: executed or saved-output
- benchmark binary and range metadata
- GPU count
- every parsed NCCL result row
- sample count
- mean and peak algorithm bandwidth
- mean and peak bus bandwidth
- largest tested message size

The baseline is attached to the campaign report, not the hashed campaign specification. This keeps existing schema-v1 manifests valid while preserving the measured communication environment in the campaign artifact.

## Software/container provenance

Campaign provenance now additionally supports:

- CUDA toolkit version
- NCCL version
- container image
- container image digest

CUDA toolkit version is detected from `nvcc --version` when possible.

NCCL version is taken from `NCCL_VERSION` when present or from installed NVIDIA NCCL Python package metadata when available.

Operators can set exact values explicitly:

```bash
--cuda-toolkit-version 13.0
--nccl-version 2.28.3
--container-image registry.example/train:2026-10
--container-image-digest sha256:...
```

Container metadata can also be supplied through:

```text
AGO_CONTAINER_IMAGE
AGO_CONTAINER_IMAGE_DIGEST
```

The campaign resume compatibility guard includes these fields, so a changed CUDA toolkit, NCCL version, image or digest blocks resume unless the operator uses the existing explicit provenance-change override.

## Trial-level export

The export surface intentionally emits one row per optimizer trial rather than one row per replicate.

That preserves unsuccessful and non-selected configurations, which is important for performance modeling and paper-quality analysis.

```bash
agoctl campaign export <campaign-id> \
  --format csv \
  --output results.csv
```

Supported formats:

- `csv`
- `jsonl`

Each row contains:

- campaign ID, name and group
- replicate index
- optimization ID
- trial index
- selected-best flag
- feasibility
- configuration label
- tensor/pipeline/data parallel sizes
- batch size
- concurrency
- throughput
- p99 latency
- energy
- GPU utilization
- memory pressure
- duration and exit code
- adapter-native metrics as `extra_*`
- driver version
- CUDA runtime version
- CUDA toolkit version
- NCCL version
- container image/digest
- GPU count/models
- campaign manifest fingerprint
- NCCL mean/peak algorithm and bus bandwidth

## H100 vs H200 workflow

A reproducible comparison can now use:

1. run the H100 campaign with an NCCL baseline
2. export the H100 trial table
3. run the H200 campaign with the same hash-verified manifest and an H200 NCCL baseline
4. export the H200 trial table
5. use `agoctl campaign compare` for the built-in statistical view
6. combine the two CSV/JSONL files for deeper modeling or publication figures

This separates model/workload behavior from communication-fabric context and records enough software/container provenance to make the comparison auditable.

## Safety

NCCL execution is opt-in.

The command only runs when the operator supplies `--nccl-baseline`; otherwise no communication benchmark is launched.

Campaign export only reads stored campaign/optimization artifacts and writes the requested output file. It does not mutate Kubernetes, Slurm, GPU settings or campaign results.
