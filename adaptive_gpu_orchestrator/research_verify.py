"""Independent offline verification of source evidence referenced by a research report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from adaptive_gpu_orchestrator.campaigns import CampaignStore
from adaptive_gpu_orchestrator.modeling.evidence_fingerprint import (
    fingerprint_saved_evidence,
)
from adaptive_gpu_orchestrator.optimizer import OptimizationStore


def verify_report(
    report_path: Path,
    *,
    campaign_store: CampaignStore,
    optimization_store: OptimizationStore,
    checksum_path: Path | None = None,
) -> dict[str, object]:
    """Verify JSON checksum (when provided) and saved evidence fingerprints.

    Not a cryptographic signature or proof of genuine benchmark measurements.
    """
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("kind") != "gpu-hardware-research-report":
        raise ValueError("invalid research report kind")
    manifest = report.get("evidence_manifest")
    if manifest is None:
        raise ValueError("report lacks source evidence manifest")
    if not isinstance(manifest, dict):
        raise TypeError("source evidence manifest must be an object")
    if manifest.get("schema_version") != 1 or manifest.get("hash_algorithm") != "sha256":
        raise ValueError("unsupported source evidence manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("empty or malformed source evidence manifest")
    campaign_ids = [item["campaign_id"] for item in sources]
    if any(not isinstance(cid, str) or not cid for cid in campaign_ids):
        raise ValueError("invalid campaign identifier")
    if len(set(campaign_ids)) != len(campaign_ids):
        raise ValueError("duplicate campaigns in evidence manifest")
    campaigns = [campaign_store.load(identifier) for identifier in campaign_ids]
    actual = fingerprint_saved_evidence(campaigns, optimization_store=optimization_store)
    if actual != manifest:
        raise ValueError("source evidence fingerprint mismatch")
    checksum_verified = False
    if checksum_path is not None:
        line = checksum_path.read_text(encoding="utf-8").strip()
        fields = line.split("  ", 1)
        if len(fields) != 2 or fields[1] != report_path.name:
            raise ValueError("invalid report checksum manifest")
        expected = fields[0]
        digest = hashlib.sha256(raw).hexdigest()
        if expected != digest:
            raise ValueError("report JSON checksum mismatch")
        checksum_verified = True
    return {
        "valid": True,
        "campaign_count": len(campaign_ids),
        "manifest_sha256": actual["manifest_sha256"],
        "report_checksum_verified": checksum_verified,
        "limitations": "Fingerprint consistency does not authenticate measured hardware or benchmark provenance.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ago-research-verify",
        description="Verify saved source evidence against an offline research report; no GPU runs.",
    )
    parser.add_argument("report", type=Path, help="Existing research-report.json")
    parser.add_argument("--campaign-store", default=".ago/campaigns")
    parser.add_argument("--optimization-store", default=".ago/optimizations")
    parser.add_argument("--signature", type=Path, help="Optional detached Ed25519 signature")
    parser.add_argument("--public-key", type=Path, help="Trusted PEM Ed25519 public key")
    parser.add_argument("--expected-key-id", help="Expected independently trusted signer key ID")
    parser.add_argument("--checksum", type=Path,
                        help="Optional explicit research-report.sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (args.signature is None) != (args.public_key is None):
            raise ValueError("--signature and --public-key must be specified together")
        if args.expected_key_id is not None and args.signature is None:
            raise ValueError("--expected-key-id requires --signature")
        result = verify_report(
            args.report,
            campaign_store=CampaignStore(Path(args.campaign_store)),
            optimization_store=OptimizationStore(Path(args.optimization_store)),
            checksum_path=args.checksum,
        )
        if args.signature is not None:
            from adaptive_gpu_orchestrator.research_attestation import (
                verify_research_signature,
            )

            result["signature"] = verify_research_signature(
                args.report, args.signature, args.public_key,
                expected_key_id=args.expected_key_id,
            )
    except (OSError, ValueError, TypeError, KeyError, ImportError, json.JSONDecodeError) as exc:
        print(f"research verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
