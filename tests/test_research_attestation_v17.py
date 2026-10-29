"""Ed25519 detached report signature tests using temporary in-memory keys."""
import json

import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_research_disk_e2e_v17 import build_saved_experiments

from adaptive_gpu_orchestrator.research_attestation import (
    sign_research_report,
    verify_research_signature,
)
from adaptive_gpu_orchestrator.research_cli import main as generate
from adaptive_gpu_orchestrator.research_sign_cli import main as sign_cli
from adaptive_gpu_orchestrator.research_verify import main as verify_cli


def keys(tmp_path):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public_path.write_bytes(private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    return private_path, public_path


def test_signature_roundtrip_wrong_key_and_tampering(tmp_path):
    private_path, public_path = keys(tmp_path)
    report = tmp_path / "research-report.json"
    report.write_text('{"kind":"fixture"}\n', encoding="utf-8")
    signature = tmp_path / "report.sig.json"
    sign_research_report(report, private_path, output_path=signature, key_id="lab-key-1")
    status = verify_research_signature(
        report, signature, public_path, expected_key_id="lab-key-1"
    )
    assert status["valid"] is True
    assert status["key_id"] == "lab-key-1"
    with pytest.raises(ValueError, match="expected signer"):
        verify_research_signature(
            report, signature, public_path, expected_key_id="another-key"
        )
    wrong_key = Ed25519PrivateKey.generate().public_key()
    public_path.write_bytes(wrong_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    with pytest.raises(ValueError, match="signature verification failed"):
        verify_research_signature(report, signature, public_path)
    report.write_text('{"kind":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_research_signature(report, signature, public_path)


def test_full_cli_signature_verification(tmp_path, capsys):
    campaigns, optimizations, ids, hardware = build_saved_experiments(tmp_path)
    out = tmp_path / "reports"
    assert generate([
        *ids, "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--hardware-json", str(hardware), "--output-dir", str(out),
        "--iterations", "20",
    ]) == 0
    private_path, public_path = keys(tmp_path)
    sig = tmp_path / "report.sig.json"
    report = out / "research-report.json"
    assert sign_cli([
        str(report), "--private-key", str(private_path), "--key-id", "research-v1",
        "--output", str(sig),
    ]) == 0
    capsys.readouterr()
    argv = [
        str(report), "--campaign-store", str(campaigns.root),
        "--optimization-store", str(optimizations.root),
        "--checksum", str(out / "research-report.sha256"),
        "--signature", str(sig), "--public-key", str(public_path),
        "--expected-key-id", "research-v1",
    ]
    assert verify_cli(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] and result["signature"]["valid"]
    assert result["signature"]["key_id"] == "research-v1"
    assert verify_cli(argv[:-2] + ["--expected-key-id", "wrong"]) == 2
