# Verify an existing research report against saved evidence

After creating a report using `ago-research-report`, run:

```bash
ago-research-verify research-output/research-report.json \
  --campaign-store .ago/campaigns \
  --optimization-store .ago/optimizations \
  --checksum research-output/research-report.sha256
```

The verifier loads the report's embedded campaign and optimization manifest,
retrieves the corresponding saved records, and recomputes canonical SHA-256
fingerprints. If supplied, the JSON checksum file additionally detects
modifications to report fields outside the evidence manifest. It returns
JSON with `valid=true` and exit code 0 on success, or exit code 2 and a
diagnostic on failure.

Use `--checksum` when checking whether **the entire original JSON report**
is unchanged; without it, only the saved source-evidence manifest is
verified. A checksum is neither signed nor authenticated. A malicious
party controlling both report and checksum can replace both; fingerprints
cannot certify that the original benchmark measurements were genuine.
The command does not trigger GPU work, model fitting, or deployment.
