# Reproducible Evidence Bundles

v1.1 turns an autopilot decision and its supporting artifacts into one portable, integrity-checkable JSON bundle.

## Why bundles

A decision can depend on multiple persisted artifacts:

- baseline optimization
- candidate optimization
- statistical study
- telemetry correlation
- per-GPU time-series analysis
- autopilot validation, policy reasoning, deployment plans and rollback plan

Those files normally live in separate `.ago` stores. An evidence bundle embeds the relevant artifacts so the decision can be reviewed or archived without relying on the original filesystem layout.

## Artifact references

Bundle creation starts from a saved autopilot report.

The builder verifies that:

- the supplied baseline optimization ID matches the autopilot baseline ID
- the supplied candidate optimization ID matches the autopilot candidate ID
- any study ID referenced by the autopilot report matches the supplied study
- any correlation ID referenced by the autopilot report matches the supplied correlation
- any time-series ID referenced by the autopilot report matches the supplied time-series report

Unexpected evidence is rejected as well. This prevents a bundle from silently attaching unrelated artifacts.

## Canonical SHA-256 hashing

Every embedded artifact is serialized using canonical JSON:

- object keys sorted
- compact separators
- UTF-8 encoding
- non-finite JSON numbers rejected

The SHA-256 hash is computed over those canonical bytes.

The bundle itself is also hashed over its unsigned canonical representation, including:

- bundle ID
- autopilot report ID
- all artifact metadata, payloads and artifact hashes
- runtime provenance

The final bundle SHA-256 is stored separately.

## Verification

`verify_bundle()` performs two checks:

1. recompute every artifact hash from the embedded payload
2. recompute the complete bundle hash

Verification does not need access to the original optimization, study, correlation, time-series or autopilot stores.

CLI:

```bash
agoctl bundle verify <bundle-id>
```

Exit codes:

- `0`: integrity verified
- `5`: artifact or bundle fingerprint mismatch
- `2`: missing bundle or invalid input

## Runtime provenance

Bundle creation captures:

- Python version
- Python implementation
- operating-system/platform description
- machine architecture
- installed package version when available
- git commit when the working tree is a git checkout
- user-supplied tags

Environment capture is intentionally restricted to an allowlist:

- `CUDA_VISIBLE_DEVICES`
- `NVIDIA_VISIBLE_DEVICES`
- `NCCL_DEBUG`
- `NCCL_SOCKET_IFNAME`
- `OMP_NUM_THREADS`

The bundle does not enumerate the full process environment, which avoids accidentally collecting credentials or unrelated secrets.

## User-defined tags

Tags provide experiment context without changing any source artifact.

Example:

```bash
agoctl bundle create \
  --autopilot abc123 \
  --tag hardware=h200-sxm \
  --tag model=llama-70b \
  --tag cuda=13.0
```

## Stored format

Bundles are saved under:

```text
.ago/bundles/
```

The JSON file contains:

- bundle identity and bundle SHA-256
- autopilot report ID
- runtime provenance
- embedded evidence artifacts
- artifact kind, source identifier, payload and SHA-256

## CLI

Create:

```bash
agoctl bundle create --autopilot <decision-id>
```

List:

```bash
agoctl bundle list
```

Show with integrity status:

```bash
agoctl bundle show <bundle-id>
```

Verify:

```bash
agoctl bundle verify <bundle-id>
```

## Security and safety boundary

Evidence bundles are passive local artifacts.

Creating, showing, listing or verifying a bundle does not:

- contact Kubernetes
- submit Slurm jobs
- modify scheduler state
- mutate the embedded source artifacts
- collect arbitrary environment variables

The v1.0 recommend-only scheduler boundary remains unchanged.

## Research use

A bundle can be archived alongside a paper, internal benchmark report or experiment notebook as a machine-readable record of:

- what the optimizer selected
- what evidence was considered
- what the autopilot decided
- what policy was applied
- how rollback was defined
- what runtime generated the bundle
- whether the artifact remains bit-for-bit consistent with its stored fingerprints

## Future extensions

- detached cryptographic signatures
- SLSA-style build provenance
- OCI artifact export
- artifact schema/version identifiers
- source-code tree fingerprints
- container-image digests
- CUDA/driver/NCCL version provenance
- bundle comparison and regression reports
