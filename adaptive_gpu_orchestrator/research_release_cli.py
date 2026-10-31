"""Create or verify a versioned offline research release manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ago-research-release")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Create a manifest from existing reports")
    create.add_argument("directory", type=Path)
    create.add_argument("--release-id", required=True)
    create.add_argument("--source-revision", required=True)
    verify = sub.add_parser("verify", help="Verify a research release")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--expected-release-id")
    verify.add_argument("--expected-revision")
    verify.add_argument("--expected-manifest-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from adaptive_gpu_orchestrator.research_release import (
            create_release_manifest,
            verify_release_manifest,
        )

        if args.command == "create":
            path = create_release_manifest(
                args.directory,
                release_id=args.release_id,
                source_revision=args.source_revision,
            )
            result = {"manifest": str(path)}
        else:
            result = verify_release_manifest(
                args.manifest,
                expected_release_id=args.expected_release_id,
                expected_revision=args.expected_revision,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"research release failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
