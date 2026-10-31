"""Versioned offline research releases with independently checkable file hashes.

A release manifest is not itself signed. For authenticity, consumers must
compare its digest against an independently trusted record or sign it
separately using a trusted key distribution process.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_FILES = ("research-report.json", "research-report.html", "research-report.md",
          "heldout-folds.csv", "research-report.sha256",
          "research-report.sig.json")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_release_manifest(
    directory: str | Path,
    *,
    release_id: str,
    source_revision: str,
) -> Path:
    """Fingerprint a fixed allowlist of report artifacts and source revision."""
    if not _RELEASE_ID.fullmatch(release_id):
        raise ValueError("invalid release identifier")
    if not _REVISION.fullmatch(source_revision):
        raise ValueError("source_revision must be a full 40-character lowercase Git SHA")
    root = Path(directory)
    required = _FILES[:5]
    for filename in required:
        if not (root / filename).is_file():
            raise FileNotFoundError(f"release artifact missing: {filename}")
    files = {
        filename: _hash(root / filename)
        for filename in _FILES if (root / filename).is_file()
    }
    report = json.loads((root / "research-report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("kind") != "gpu-hardware-research-report":
        raise ValueError("invalid research report kind")
    evidence = report.get("evidence_manifest")
    if evidence is None:
        raise ValueError("release requires source evidence manifest")
    if not isinstance(evidence, dict):
        raise TypeError("source evidence manifest must be an object")
    if not isinstance(evidence.get("manifest_sha256"), str):
        raise ValueError("release requires source evidence manifest digest")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "research-release-manifest",
        "release_id": release_id,
        "source_revision": source_revision,
        "source_evidence_sha256": evidence["manifest_sha256"],
        "files_sha256": files,
    }
    destination = root / "research-release.json"
    destination.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def verify_release_manifest(
    path: str | Path,
    *,
    expected_release_id: str | None = None,
    expected_revision: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Check source evidence link and all referenced file bytes, fail closed."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") != "research-release-manifest":
        raise ValueError("invalid research release manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported release manifest version")
    release = manifest.get("release_id")
    revision = manifest.get("source_revision")
    if not isinstance(release, str) or not _RELEASE_ID.fullmatch(release):
        raise ValueError("invalid release identifier")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError("invalid source revision")
    if expected_release_id is not None and release != expected_release_id:
        raise ValueError("unexpected research release identifier")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("unexpected source revision")
    digest = _hash(Path(path))
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("release manifest digest mismatch")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not set(_FILES[:5]).issubset(files):
        raise ValueError("release is missing mandatory artifact fingerprints")
    if not set(files).issubset(_FILES):
        raise ValueError("release contains unrecognized artifact names")
    root = Path(path).parent
    for filename, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"malformed fingerprint for {filename}")
        if not (root / filename).is_file() or _hash(root / filename) != expected:
            raise ValueError(f"release artifact mismatch: {filename}")
    report = json.loads((root / "research-report.json").read_text(encoding="utf-8"))
    evidence = report.get("evidence_manifest") if isinstance(report, dict) else None
    if not isinstance(evidence, dict) or evidence.get("manifest_sha256") != manifest.get(
        "source_evidence_sha256"
    ):
        raise ValueError("source evidence link mismatch")
    return {
        "valid": True, "release_id": release, "source_revision": revision,
        "manifest_sha256": digest, "verified_files": len(files),
        "limitations": "Unsigned manifest is change-detectable only when its digest is independently trusted.",
    }
