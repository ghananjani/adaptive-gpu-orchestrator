from __future__ import annotations

import argparse
import json
import sys

from adaptive_gpu_orchestrator.benchmarks import NCCLBenchmark, parse_nccl_output
from adaptive_gpu_orchestrator.doctor import Doctor


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=False))


def _plain_snapshot(snapshot) -> None:
    print(f"Host: {snapshot.hostname}")
    print(f"Driver: {snapshot.driver_version or 'unknown'}  CUDA: {snapshot.cuda_version or 'unknown'}")
    print(f"GPUs: {len(snapshot.gpus)}")
    for gpu in snapshot.gpus:
        mem = f"{gpu.memory_used_mib}/{gpu.memory_total_mib} MiB" if gpu.memory_total_mib else "unknown"
        print(f"  GPU {gpu.index}: {gpu.name} | util={gpu.utilization_gpu_pct}% | memory={mem}")
    for warning in snapshot.warnings:
        print(f"warning: {warning}", file=sys.stderr)


def _plain_topology(snapshot) -> None:
    if not snapshot.topology:
        print("No GPU-to-GPU topology links discovered.")
        return
    for link in sorted(snapshot.topology, key=lambda x: (x.source, x.target)):
        print(f"GPU{link.source} <-> GPU{link.target}: {link.kind} (weight={link.weight})")


def _plain_report(report) -> None:
    print(f"Bottleneck: {report.bottleneck.value}")
    print(f"Severity: {report.severity}")
    print(report.summary)
    if report.evidence:
        print("\nEvidence:")
        for item in report.evidence:
            print(f"  - {item.signal}: {item.value} — {item.reason}")
    if report.recommendations:
        print("\nRecommendations:")
        for item in report.recommendations:
            print(f"  - [{item.priority}] {item.action}: {item.rationale}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agoctl", description="Adaptive GPU Orchestrator for AI Workloads")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="discover GPUs and current utilization")
    sub.add_parser("topology", help="show discovered GPU-to-GPU topology")
    doctor_cmd = sub.add_parser("doctor", help="diagnose high-confidence bottlenecks")
    doctor_cmd.add_argument("--nccl-output", help="path to saved nccl-tests output")

    nccl = sub.add_parser("benchmark-nccl", help="run nccl-tests all_reduce_perf")
    nccl.add_argument("--binary", default="all_reduce_perf")
    nccl.add_argument("--begin", default="8M")
    nccl.add_argument("--end", default="1G")
    nccl.add_argument("--factor", type=int, default=2)
    nccl.add_argument("--gpus", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    doctor = Doctor()

    if args.command == "inspect":
        snapshot = doctor.inspect()
        _print_json(snapshot.to_dict()) if args.json else _plain_snapshot(snapshot)
        return 0

    if args.command == "topology":
        snapshot = doctor.inspect()
        _print_json([x.to_dict() for x in snapshot.topology]) if args.json else _plain_topology(snapshot)
        return 0

    if args.command == "doctor":
        nccl_results = None
        if args.nccl_output:
            with open(args.nccl_output, "r", encoding="utf-8") as handle:
                nccl_results = parse_nccl_output(handle.read())
        report = doctor.diagnose(nccl=nccl_results)
        _print_json(report.to_dict()) if args.json else _plain_report(report)
        return 0

    if args.command == "benchmark-nccl":
        results = NCCLBenchmark(args.binary).run(args.begin, args.end, args.factor, args.gpus)
        if args.json:
            _print_json([x.to_dict() for x in results])
        else:
            for row in results:
                print(f"{row.size_bytes:>12} bytes  algbw={row.algbw_gbps:>8.2f} GB/s  busbw={row.busbw_gbps:>8.2f} GB/s")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
