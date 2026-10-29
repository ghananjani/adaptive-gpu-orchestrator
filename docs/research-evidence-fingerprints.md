# Canonical source evidence fingerprints

The `ago-research-report` command now includes an `evidence_manifest` in
`research-report.json`. It fingerprints full serialized CampaignReport
records and every referenced completed OptimizationResult, then hashes the
sorted manifest entries. The ordering and compact JSON representation are
stable across runs, independent of file modification time or local path.

The manifest allows auditors to detect changed source records, including
metadata changes that may not affect a fitted performance metric. Every
campaign and optimization ID and individual hash are present for tracing.

Hashes provide **change detection only**: they are not signatures and cannot
authenticate GPU measurements, benchmark protocol claims or upstream sources.
The inputs must still have independently confirmed provenance. The report
itself retains a SHA-256 file checksum, which now covers the embedded
source-evidence manifest. No GPUs are run.
