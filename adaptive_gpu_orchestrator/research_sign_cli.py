"""Offline detached report signing CLI; no private key is generated or stored."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ago-research-sign")
    parser.add_argument("report", type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        from adaptive_gpu_orchestrator.research_attestation import sign_research_report

        output = sign_research_report(
            args.report, args.private_key, output_path=args.output, key_id=args.key_id
        )
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f"research signing failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"signature": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
