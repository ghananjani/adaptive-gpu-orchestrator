# Campaign Manifests + Resume

v1.4 makes experiment campaigns declarative, fingerprinted, reproducible and resumable.

## Manifest identity

A campaign manifest contains a schema version, the complete `CampaignSpec`, and a SHA-256 fingerprint calculated over canonical JSON.

```json
{
  "schema_version": 1,
  "spec": {
    "name": "h100-llama70b",
    "group": "h100",
    "repeats": 5,
    "available_gpus": 8,
    "candidates": [],
    "objective": "throughput",
    "constraints": {},
    "adapter": "vllm",
    "benchmark_command": ["python", "benchmark.py"],
    "timeout_s": 120.0,
    "randomize_candidates": true,
    "random_seed": 20261016,
    "tags": {"hardware": "h100"}
  },
  "sha256": "..."
}
```

Manifest loading verifies both the schema version and SHA-256. A changed campaign field therefore requires a new fingerprint.

## Export and replay

Export the exact spec from a saved campaign:

```bash
agoctl campaign manifest <campaign-id> \
  --output experiments/h100-llama70b.json
```

Replay it:

```bash
agoctl campaign run --manifest experiments/h100-llama70b.json
```

Manifest mode deliberately rejects benchmark-command and tag overrides. Reproducible changes should create a new manifest.

## Deterministic candidate randomization

Candidate execution order can bias benchmark results when thermal state, caches or system load drift during a search.

Enable deterministic per-replicate randomization with:

```bash
--randomize-candidates --random-seed 20261016
```

A seed is mandatory when randomization is enabled.

For replicate `r`, the current implementation seeds the shuffle with:

```text
random_seed + r
```

The original first candidate remains the optimizer baseline; only execution order changes.

Each `CampaignRun` stores the actual candidate labels in `candidate_order`.

## Resume

Resume an incomplete campaign:

```bash
agoctl campaign resume <campaign-id>
```

Completed replicates are preserved and are not executed again. Replicates with `error`, `no_feasible_candidate`, or no run record are eligible for another attempt.

After resume, the one-group statistical study is rebuilt from all completed optimization IDs.

## Manifest guard during resume

New v1.4 campaigns store `manifest_sha256` in the campaign report.

Before resume, the engine recomputes the fingerprint from the stored specification. A mismatch blocks resume.

v1.3 campaign files remain compatible. When a v1.3 campaign without a fingerprint is resumed, v1.4 derives and records the fingerprint from its existing specification.

## Provenance guard

Resume captures current provenance and compares a static research-relevant signature against the original campaign:

- NVIDIA driver version
- CUDA version
- GPU model
- GPU count
- GPU total memory
- GPU compute capability
- tracked ML package versions

Dynamic values such as utilization, power and temperature are intentionally excluded from the compatibility check.

A mismatch blocks resume by default.

## Explicit environment-change override

An operator may intentionally continue under changed hardware or software:

```bash
agoctl campaign resume <campaign-id> \
  --allow-provenance-change
```

The new provenance snapshot is appended to `resume_provenance`, preserving an audit trail that the campaign spans multiple environments.

## Benchmark environment

In addition to existing campaign variables, v1.4 injects:

```text
AGO_CAMPAIGN_MANIFEST_SHA256
```

Benchmark-native logs can therefore identify the exact manifest.

## Backward compatibility

Old v1.3 JSON defaults cleanly:

```text
randomize_candidates = false
random_seed = null
candidate_order = []
manifest_sha256 = null
resume_count = 0
resume_provenance = []
```

## Safety boundary

Resume executes only the benchmark command already stored in the campaign specification.

It does not:

- alter scheduler state
- submit Kubernetes or Slurm workloads
- change GPU clocks or power caps
- change MIG state
- run in the background
- rerun completed replicates

## Research workflow

```text
version-controlled manifest
        ↓
campaign run
        ↓
partial failure
        ↓
campaign resume
        ↓
statistical study
        ↓
campaign comparison
        ↓
autopilot / evidence bundle
```

This gives experiment results a stable, reviewable specification identity instead of relying on shell history.
