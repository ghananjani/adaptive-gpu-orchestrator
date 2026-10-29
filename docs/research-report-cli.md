# One-command offline GPU research report

Install this repository in your Python 3.10+ environment and invoke:

```bash
ago-research-report \
  H100-run-a H100-run-b H200-run-a H200-run-b L40S-run-a L40S-run-b \
  --campaign-store .ago/campaigns \
  --optimization-store .ago/optimizations \
  --hardware-json measured-hardware.json \
  --output-dir research-output \
  --iterations 200 --seed 17
```

`measured-hardware.json` must map **exact** `gpu_models` values recorded
in the saved trial rows to verified physical descriptors:

```json
{
  "EXACT_PLATFORM_SIGNATURE_FROM_TRIALS": {
    "gpu_count": 8,
    "memory_gib_per_gpu": 80,
    "memory_bandwidth_gbps_per_gpu": 3000,
    "compute_tflops_per_gpu": 60,
    "interconnect_bandwidth_gbps": 300
  }
}
```

**All example numbers above are illustrative placeholders, not verified
specifications or measurements.** Replace them with independently confirmed
values for each actual platform before evaluation. Three distinct platforms,
at least two independent campaigns per platform, matched workload/protocol
tags, consistent software provenance, and stored optimization trials are
required. The command fails rather than using inferred/default GPU hardware.

The command runs strict saved-evidence validation, leakage/design audit,
leave-one-platform-out paired model comparison and training-campaign refit
bootstrap. It creates `research-report.html`, `research-report.md`,
`research-report.json`, `heldout-folds.csv` and
`research-report.sha256` in the output directory. It never executes GPU
benchmarks or mutates production services. By default, inconsistent software
provenance fails closed; the explicit `--allow-software-mismatch` research
override must be disclosed in the resulting report.

This bootstrap interval is conditional on the held-out hardware and test
observations; it is not a calibrated confidence interval for new GPU families.
Use `ago-research-report --help` for available flags.
