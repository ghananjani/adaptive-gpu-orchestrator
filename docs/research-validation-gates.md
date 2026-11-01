# Held-out benchmark validation gate

The `ago-research-gate` command evaluates **existing** saved research reports
against explicit acceptance criteria without fitting models, running GPUs,
or deploying anything. Generate the report with `ago-research-report` first.

```bash
ago-research-gate research-output/research-report.json \
  --min-platforms 3 \
  --min-test-samples-per-platform 6 \
  --min-bootstrap-valid-fraction 0.9 \
  --max-conditioned-rmse 15 \
  --min-fold-rmse-reduction-pct 0
```

All thresholds are study-specific and must be chosen before evaluating
results, based on benchmark units and intended use. Zero is not a universal
RMSE or model-performance target. The gate checks each held-out hardware
platform separately rather than hiding poor transfers behind a pooled mean,
and it rejects insufficient valid bootstrap draws.

**Exit codes:** 0 = all thresholds passed, 3 = valid report but acceptance
failed, 2 = invalid input/report. JSON output includes explicit failures
and the exact thresholds used. Negative RMSE reduction indicates the
hardware-conditioned model performed worse than the config-only baseline.

This is an offline research-data acceptance gate, **not** a proof of
unseen-platform generalization or a production scheduler approval. The
test fixture contains synthetic measurements only. Independently confirm
report provenance with `ago-research-verify`, and gather real measured GPU
evidence before making hardware-performance claims.
