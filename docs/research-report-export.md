# Offline research report exports

Run the validated saved-campaign refitting pipeline and export a stakeholder
report **without** any GPU benchmarks or production-side changes.

```python
from adaptive_gpu_orchestrator.modeling import (
    evaluate_saved_refitting,
    write_research_report,
)

result = evaluate_saved_refitting(
    campaigns,
    optimization_store=optimization_store,
    measured_hardware=measured_hardware,
    iterations=200, seed=17,
)
paths = write_research_report(result, "research-output")
print(paths)
```

Generated files:
- `research-report.html`: printable, self-contained offline comparison summary
- `research-report.md`: textual summary and methodological cautions
- `research-report.json`: full input evidence, campaign audit, and refitting result
- `heldout-folds.csv`: original per-hardware fold errors
- `research-report.sha256`: checksum of JSON bytes, not a cryptographic signature

The outputs summarize baseline vs hardware-conditioned pooled held-out RMSE,
per-platform fold errors, training-campaign refitting sensitivity bounds,
valid/invalid bootstrap draws, exclusion counts, workload tags, source
trial hash and all audit/provenance warnings.

**Important:** Reports do not generate new measurements, establish calibration
or prove one model is superior. The recorded uncertainty is conditional on
observed hardware and held-out tests. The report generator is dependency-free
and deliberately avoids remote charting assets or HTML script execution.
Use the source JSON to inspect software/measurement provenance. The JSON SHA
fingerprint is only a reproducibility aid, not evidence of data authenticity.
