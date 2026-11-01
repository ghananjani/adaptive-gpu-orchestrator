"""Command-line acceptance checks for persisted research measurements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_gpu_orchestrator.modeling.validation_gate import (
    ValidationPolicy,
    evaluate_validation_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ago-research-gate")
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-platforms", type=int, default=3)
    parser.add_argument("--min-test-samples-per-platform", type=int, default=1)
    parser.add_argument("--max-conditioned-rmse", type=float)
    parser.add_argument("--min-fold-rmse-reduction-pct", type=float)
    parser.add_argument("--min-bootstrap-valid-fraction", type=float, default=0.9)
    args = parser.parse_args(argv)
    try:
        policy = ValidationPolicy(
            min_platforms=args.min_platforms,
            min_test_samples_per_platform=args.min_test_samples_per_platform,
            max_conditioned_rmse=args.max_conditioned_rmse,
            min_fold_rmse_reduction_pct=args.min_fold_rmse_reduction_pct,
            min_bootstrap_valid_fraction=args.min_bootstrap_valid_fraction,
        )
        report = json.loads(args.report.read_text(encoding="utf-8"))
        result = evaluate_validation_gate(report, policy)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"research gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
