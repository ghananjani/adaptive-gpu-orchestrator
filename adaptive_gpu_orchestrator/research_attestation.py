"""Detached Ed25519 signatures for immutable-by-policy research report artifacts.

Uses public-key cryptography, never stores private keys in reports.
Authenticity depends on independent trusted public-key distribution.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

SCHEMA_VERSION = 1
DOMAIN = b"adaptive-gpu-orchestrator:research-report:v1\x00"


def _message(report_bytes: bytes) -> bytes:
    return DOMAIN + hashlib.sha256(report_bytes).digest()


def sign_research_report(
    report_path: str | Path,
    private_key_path: str | Path,
    *,
    output_path: str | Path,
    key_id: str,
) -> Path:
    """Sign report content using an existing PEM Ed25519 private key."""
    if not key_id.strip():
        raise ValueError("key_id cannot be empty")
    key = serialization.load_pem_private_key(
        Path(private_key_path).read_bytes(), password=None
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("signing key must be Ed25519")
    raw = Path(report_path).read_bytes()
    signed = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "Ed25519",
        "domain": "adaptive-gpu-orchestrator:research-report:v1",
        "key_id": key_id,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "signature_base64": base64.b64encode(key.sign(_message(raw))).decode("ascii"),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(signed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def verify_research_signature(
    report_path: str | Path,
    signature_path: str | Path,
    public_key_path: str | Path,
    *,
    expected_key_id: str | None = None,
) -> dict[str, Any]:
    """Verify report signature against an independently trusted PEM public key."""
    record = json.loads(Path(signature_path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError("signature record must be an object")
    if (record.get("schema_version") != SCHEMA_VERSION
            or record.get("algorithm") != "Ed25519"
            or record.get("domain") != "adaptive-gpu-orchestrator:research-report:v1"):
        raise ValueError("unsupported signature record")
    key_id = record.get("key_id")
    if not isinstance(key_id, str) or not key_id.strip():
        raise ValueError("invalid signature key ID")
    if expected_key_id is not None and key_id != expected_key_id:
        raise ValueError("signature key ID does not match expected signer")
    public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("verification key must be Ed25519")
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.get("report_sha256"):
        raise ValueError("signed report digest mismatch")
    encoded = record.get("signature_base64")
    if not isinstance(encoded, str):
        raise TypeError("signature must be base64 text")
    try:
        signature = base64.b64decode(encoded, validate=True)
        public_key.verify(signature, _message(raw))
    except (InvalidSignature, ValueError) as exc:
        raise ValueError("report signature verification failed") from exc
    return {
        "valid": True,
        "algorithm": "Ed25519",
        "key_id": key_id,
        "report_sha256": record["report_sha256"],
        "limitations": (
            "Signature authenticates the report only against the independently trusted key; "
            "it does not establish the truth of the benchmark measurements."
        ),
    }
