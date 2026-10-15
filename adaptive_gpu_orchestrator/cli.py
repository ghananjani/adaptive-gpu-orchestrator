from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_gpu_orchestrator.benchmarks import NCCLBenchmark, parse_nccl_output
from adaptive_gpu_orchestrator.doctor import Doctor
from adaptive_gpu_orchestrator.profiling import ExperimentStore, WorkloadProfiler
from adaptive_gpu_orchestrator.telemetry import (
    NVMLProvider,
    NVMLUnavailable,
    NvidiaSMIProvider,
)


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=False))


def _plain_snapshot(snapshot) -> None:
    print(f"Host: {snapshot.hostname}")
    print(
        f"Driver: {snapshot.driver_version or 'unknown'}  "
        f"CUDA: {snapshot.cuda_version or 'unknown'}"
    )
    print(f"GPUs: {len(snapshot.gpus)}")
    for gpu in snapshot.gpus:
        mem = (
            f"{gpu.memory_used_mib}/{gpu.memory_total_mib} MiB"
            if gpu.memory_total_mib
            else "unknown"
        )
        print(
            f"  GPU {gpu.index}: {gpu.name} | "
            f"util={gpu.utilization_gpu_pct}% | memory={mem}"
        )
    for warning in snapshot.warnings:
        print(f"warning: {warning}", file=sys.stderr)


def _plain_topology(snapshot) -> None:
    if not snapshot.topology:
        print("No GPU-to-GPU topology links discovered.")
        return
    for link in sorted(snapshot.topology, key=lambda x: (x.source, x.target)):
        print(
            f"GPU{link.source} <-> GPU{link.target}: "
            f"{link.kind} (weight={link.weight})"
        )


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


def _plain_profile(profile) -> None:
    summary = profile.summary()
    print(f"Experiment: {summary['id']}  {summary['name']}")
    print(
        f"Duration: {summary['duration_s']:.2f}s  "
        f"Samples: {summary['sample_count']}  "
        f"GPUs: {summary['gpu_count_max']}"
    )
    util = summary["gpu_util_mean_pct"]
    memory = summary["memory_pressure_mean_pct"]
    power = summary["power_draw_mean_w"]
    print(
        "Averages: "
        f"GPU util={util:.1f}% " if util is not None else "Averages: GPU util=unknown "
    )
    print(
        f"memory={memory:.1f}%  " if memory is not None else "memory=unknown  ",
        end="",
    )
    print(f"power={power:.1f}W" if power is not None else "power=unknown")
    if profile.command:
        print(f"Command: {' '.join(profile.command)}")
    if profile.exit_code is not None:
        print(f"Exit code: {profile.exit_code}")


def _parse_tags(values: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid tag '{value}'; expected KEY=VALUE")
        key, item = value.split("=", 1)
        if not key:
            raise ValueError("tag key cannot be empty")
        tags[key] = item
    return tags


def _provider(name: str):
    if name == "nvml":
        return NVMLProvider()
    return NvidiaSMIProvider()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agoctl",
        description="Adaptive GPU Orchestrator for AI Workloads",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="discover GPUs and current utilization")
    sub.add_parser("topology", help="show discovered GPU-to-GPU topology")

    doctor_cmd = sub.add_parser("doctor", help="diagnose high-confidence bottlenecks")
    diagnosis_input = doctor_cmd.add_mutually_exclusive_group()
    diagnosis_input.add_argument("--nccl-output", help="path to saved nccl-tests output")
    diagnosis_input.add_argument(
        "--profile",
        help="experiment id or JSON profile path to diagnose",
    )
    doctor_cmd.add_argument(
        "--store-dir",
        default=".ago/experiments",
        help="experiment store used with --profile",
    )

    nccl = sub.add_parser("benchmark-nccl", help="run nccl-tests all_reduce_perf")
    nccl.add_argument("--binary", default="all_reduce_perf")
    nccl.add_argument("--begin", default="8M")
    nccl.add_argument("--end", default="1G")
    nccl.add_argument("--factor", type=int, default=2)
    nccl.add_argument("--gpus", type=int, default=8)

    profile = sub.add_parser(
        "profile",
        help="sample GPU telemetry around a command or timed observation window",
    )
    profile.add_argument("--name", default="workload")
    profile.add_argument("--duration", type=float, default=10.0)
    profile.add_argument("--interval", type=float, default=1.0)
    profile.add_argument(
        "--provider",
        choices=("nvidia-smi", "nvml"),
        default="nvidia-smi",
    )
    profile.add_argument("--store-dir", default=".ago/experiments")
    profile.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="attach searchable metadata; may be repeated",
    )
    profile.add_argument(
        "workload_command",
        nargs=argparse.REMAINDER,
        help="command to run after --; omit for a timed telemetry-only profile",
    )

    experiments = sub.add_parser("experiments", help="inspect saved workload profiles")
    experiments.add_argument("--store-dir", default=".ago/experiments")
    experiment_sub = experiments.add_subparsers(
        dest="experiments_action",
        required=True,
    )
    experiment_sub.add_parser("list", help="list saved profiles")
    show = experiment_sub.add_parser("show", help="show a saved profile")
    show.add_argument("identifier")

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
        _print_json([x.to_dict() for x in snapshot.topology]) if args.json else _plain_topology(
            snapshot
        )
        return 0

    if args.command == "doctor":
        if args.profile:
            store = ExperimentStore(Path(args.store_dir))
            report = doctor.diagnose_profile(store.load(args.profile))
        else:
            nccl_results = None
            if args.nccl_output:
                with open(args.nccl_output, "r", encoding="utf-8") as handle:
                    nccl_results = parse_nccl_output(handle.read())
            report = doctor.diagnose(nccl=nccl_results)
        _print_json(report.to_dict()) if args.json else _plain_report(report)
        return 0

    if args.command == "benchmark-nccl":
        results = NCCLBenchmark(args.binary).run(
            args.begin,
            args.end,
            args.factor,
            args.gpus,
        )
        if args.json:
            _print_json([x.to_dict() for x in results])
        else:
            for row in results:
                print(
                    f"{row.size_bytes:>12} bytes  "
                    f"algbw={row.algbw_gbps:>8.2f} GB/s  "
                    f"busbw={row.busbw_gbps:>8.2f} GB/s"
                )
        return 0

    if args.command == "profile":
        command = list(args.workload_command)
        if command and command[0] == "--":
            command = command[1:]
        try:
            tags = _parse_tags(args.tag)
            profiler = WorkloadProfiler(provider=_provider(args.provider))
            profile = profiler.run(
                name=args.name,
                command=command or None,
                duration_s=args.duration,
                interval_s=args.interval,
                tags=tags,
            )
        except (NVMLUnavailable, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        store = ExperimentStore(Path(args.store_dir))
        path = store.save(profile)
        if args.json:
            payload = profile.to_dict()
            payload["path"] = str(path)
            _print_json(payload)
        else:
            _plain_profile(profile)
            print(f"Saved: {path}")
        return profile.exit_code or 0

    if args.command == "experiments":
        store = ExperimentStore(Path(args.store_dir))
        if args.experiments_action == "list":
            profiles = store.list()
            if args.json:
                _print_json([profile.summary() for profile in profiles])
            else:
                if not profiles:
                    print("No saved experiments.")
                for profile in profiles:
                    summary = profile.summary()
                    util = summary["gpu_util_mean_pct"]
                    util_text = f"{util:.1f}%" if util is not None else "unknown"
                    print(
                        f"{profile.id}  {profile.name}  "
                        f"{profile.duration_s:.2f}s  util={util_text}"
                    )
            return 0

        profile = store.load(args.identifier)
        _print_json(profile.to_dict()) if args.json else _plain_profile(profile)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
