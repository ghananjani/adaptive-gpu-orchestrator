# Versioned offline research releases

Create a release only after completing `ago-research-report` and optional
`ago-research-sign`. The release generator fingerprints the existing JSON,
HTML, Markdown, fold CSV, checksum and optional detached signature.

```bash
ago-research-release create research-output \
  --release-id experiment-v1.7 \
  --source-revision 0123456789abcdef0123456789abcdef01234567
ago-research-release verify research-output/research-release.json \
  --expected-release-id experiment-v1.7 \
  --expected-revision 0123456789abcdef0123456789abcdef01234567 \
  --expected-manifest-sha256 YOUR_INDEPENDENTLY_RECORDED_SHA256
```

Use the real full 40-character Git commit SHA, not the illustrative example.
The release manifest pins the report's evidence-manifest SHA-256 and the
bytes of all enumerated release files. It detects modifications and missing
files and is stable across repeated export runs.

**Trust limits:** The release manifest itself is not signed. Pass the manifest
SHA-256 obtained via an independent trusted channel to detect replacement
of the manifest and all of its listed files. Verifying a release does not
validate an Ed25519 report signature; use `ago-research-verify` with a
separately trusted public key for that. The source revision is a declared
identifier, not proof that the corresponding commit produced the report.
For immutable retention, store the bundle in versioned access-controlled
storage with your organization's retention policy. No GPUs are executed.
