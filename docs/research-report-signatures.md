# Signed offline research reports (opt-in)

Install `pip install -e '.[attestation]'` to use detached Ed25519
signatures. Generate and distribute your key pair using your organization's
approved cryptographic key-management system; **never commit a private key**
to the repository or package it with a report.

Sign an existing, already validated JSON report:

```bash
ago-research-sign research-output/research-report.json \
  --private-key /secure/path/lab-private.pem \
  --key-id lab-research-key-2026 \
  --output research-output/research-report.sig.json
```

Verify with a public key retrieved from an **independent trusted channel**:

```bash
ago-research-verify research-output/research-report.json \
  --campaign-store .ago/campaigns \
  --optimization-store .ago/optimizations \
  --checksum research-output/research-report.sha256 \
  --signature research-output/research-report.sig.json \
  --public-key /trusted/path/lab-public.pem \
  --expected-key-id lab-research-key-2026
```

Signing is separate from generating reports, so unsigned reports remain
compatible with the existing pipeline. A signature binds the exact report
bytes to the corresponding Ed25519 public key using a domain-separated
SHA-256 digest. The detached signature metadata never contains private keys.
A trusted public key MUST be pinned or independently distributed; a key ID
is descriptive and **does not itself authenticate** the signer.

Signed attestations prove possession of the signing key and detect report
changes. They do not prove benchmark measurements were genuine, provide
timestamping or revocation, or guarantee permanent immutability. Store
published reports and signatures in versioned, access-controlled storage
and establish key rotation/revocation separately.
