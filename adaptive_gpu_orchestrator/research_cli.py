"""Read-only command line research report from saved campaign evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_gpu_orchestrator.campaigns import CampaignStore
from adaptive_gpu_orchestrator.modeling import (
    HardwareFeatures,
    evaluate_saved_refitting,
    write_research_report,
)
from adaptive_gpu_orchestrator.modeling.evidence_fingerprint import fingerprint_saved_evidence
from adaptive_gpu_orchestrator.optimizer import OptimizationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ago-research-report",
        description="Validate saved GPU campaigns and export offline research reports (no GPU runs).",
    )
    parser.add_argument("campaign_ids", nargs="+", help="At least six saved campaign IDs or paths")
    parser.add_argument("--campaign-store", default=".ago/campaigns")
    parser.add_argument("--optimization-store", default=".ago/optimizations")
    parser.add_argument("--hardware-json", required=True, type=Path,
                        help="JSON mapping of exact platform signatures to measured feature objects")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metric", default="throughput_per_s")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-software-mismatch", action="store_true",
                        help="Research override; always recorded as a warning")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if len(set(args.campaign_ids)) != len(args.campaign_ids):
            raise ValueError("campaign identifiers must be unique")
        payload = json.loads(args.hardware_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError("hardware JSON must be a nonempty platform-to-features mapping")
        hardware = {
            platform: HardwareFeatures.from_dict(features)
            for platform, features in payload.items()
        }
        store = CampaignStore(Path(args.campaign_store))
        campaigns = [store.load(identifier) for identifier in args.campaign_ids]
        optimization_store = OptimizationStore(Path(args.optimization_store))
        result = evaluate_saved_refitting(
            campaigns,
            optimization_store=optimization_store,
            measured_hardware=hardware,
            metric=args.metric,
            ridge_lambda=args.ridge_lambda,
            iterations=args.iterations,
            confidence=args.confidence,
            seed=args.seed,
            allow_software_mismatch=args.allow_software_mismatch,
        )
        manifest = fingerprint_saved_evidence(
            campaigns, optimization_store=optimization_store
        )
        paths = write_research_report(result, args.output_dir, evidence_manifest=manifest)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"research report failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
