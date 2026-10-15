from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_gpu_orchestrator.benchmarks import NCCLBenchmark, parse_nccl_output
from adaptive_gpu_orchestrator.controller import (
    DeploymentPlanner,
    DeploymentPolicy,
    SchedulerBackend,
)
from adaptive_gpu_orchestrator.doctor import Doctor
from adaptive_gpu_orchestrator.optimizer import (
    CommandCanaryRunner,
    ConfigurationOptimizer,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationStore,
    ParallelismConfig,
    generate_candidates,
)
from adaptive_gpu_orchestrator.profiling import ExperimentStore, WorkloadProfiler
from adaptive_gpu_orchestrator.schedulers import KubernetesRenderer, SlurmRenderer
from adaptive_gpu_orchestrator.telemetry import (
    NvidiaSMIProvider,
    NVMLProvider,
    NVMLUnavailable,
)
from adaptive_gpu_orchestrator.validation import (
    ValidationPolicy,
    ValidationStore,
    validate_trials,
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


def _plain_optimization(result) -> None:
    feasible = sum(1 for trial in result.trials if trial.feasible)
    print(
        f"Optimization: {result.id}  objective={result.objective.value}  "
        f"trials={len(result.trials)}  feasible={feasible}"
    )
    if result.best is None:
        print("Best: none (no feasible trial had the required objective metric)")
        return

    best = result.best
    metrics = best.metrics
    print(
        f"Best: {best.config.label()}  world_size={best.config.world_size}  "
        f"score={best.score:.4f}"
    )
    if metrics.throughput_per_s is not None:
        print(f"  throughput={metrics.throughput_per_s:.3f}/s")
    if metrics.p99_latency_ms is not None:
        print(f"  p99_latency={metrics.p99_latency_ms:.3f} ms")
    if metrics.energy_joules is not None:
        print(f"  energy={metrics.energy_joules:.3f} J")
    if metrics.gpu_util_pct is not None:
        print(f"  gpu_util={metrics.gpu_util_pct:.2f}%")
    if metrics.memory_pressure_pct is not None:
        print(f"  memory_pressure={metrics.memory_pressure_pct:.2f}%")


def _plain_validation(report) -> None:
    print(
        f"Validation: {report.id}  decision={report.decision.value}  "
        f"rollback_recommended={report.rollback_recommended}"
    )
    print(
        f"Config: {report.baseline_config.label()} -> "
        f"{report.candidate_config.label()}"
    )
    if report.deltas:
        print("\nMetric deltas:")
        for name, delta in sorted(report.deltas.items()):
            print(
                f"  - {name}: {delta.baseline:.3f} -> {delta.candidate:.3f} "
                f"({delta.percent_change:+.3f}%)"
            )
    if report.reasons:
        print("\nReasons:")
        for reason in report.reasons:
            print(f"  - {reason}")


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


def _csv_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"expected comma-separated integers, got '{value}'") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError("candidate values must be positive integers")
    return values


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

    optimize = sub.add_parser(
        "optimize",
        help="benchmark candidate parallelism configurations and select the best",
    )
    optimize.add_argument("--gpus", type=int, required=True)
    optimize.add_argument("--tp", default="1,2,4,8")
    optimize.add_argument("--pp", default="1,2")
    optimize.add_argument("--dp", default="1")
    optimize.add_argument("--batch", default="1,2,4,8")
    optimize.add_argument("--concurrency", default="1,8")
    optimize.add_argument(
        "--objective",
        choices=[item.value for item in OptimizationObjective],
        default=OptimizationObjective.THROUGHPUT.value,
    )
    optimize.add_argument("--exact-world-size", action="store_true")
    optimize.add_argument("--max-candidates", type=int, default=64)
    optimize.add_argument("--timeout", type=float, default=120.0)
    optimize.add_argument("--max-p99-ms", type=float)
    optimize.add_argument("--max-energy-joules", type=float)
    optimize.add_argument("--min-throughput", type=float)
    optimize.add_argument("--max-memory-pressure", type=float)
    optimize.add_argument("--store-dir", default=".ago/optimizations")
    optimize.add_argument("--dry-run", action="store_true")
    optimize.add_argument(
        "benchmark_command",
        nargs=argparse.REMAINDER,
        help="benchmark command after --; must emit one JSON metrics object",
    )

    optimizations = sub.add_parser(
        "optimizations",
        help="inspect saved configuration-optimization runs",
    )
    optimizations.add_argument("--store-dir", default=".ago/optimizations")
    optimization_sub = optimizations.add_subparsers(
        dest="optimizations_action",
        required=True,
    )
    optimization_sub.add_parser("list", help="list saved optimization runs")
    optimization_show = optimization_sub.add_parser("show", help="show one run")
    optimization_show.add_argument("identifier")

    plan = sub.add_parser(
        "plan",
        help="render a recommend-only scheduler plan from a configuration",
    )
    plan.add_argument(
        "--backend",
        choices=[item.value for item in SchedulerBackend],
        required=True,
    )
    plan.add_argument("--available-gpus", type=int, required=True)
    plan.add_argument("--optimization")
    plan.add_argument("--optimization-store", default=".ago/optimizations")
    plan.add_argument("--tp", type=int, default=1)
    plan.add_argument("--pp", type=int, default=1)
    plan.add_argument("--dp", type=int, default=1)
    plan.add_argument("--batch", type=int, default=1)
    plan.add_argument("--concurrency", type=int, default=1)
    plan.add_argument("--max-gpus", type=int)
    plan.add_argument("--require-exact-gpu-count", action="store_true")
    plan.add_argument(
        "--node-selector",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    plan.add_argument("--image")
    plan.add_argument("--job-name", default="ago-workload")
    plan.add_argument("--nodes", type=int, default=1)
    plan.add_argument("--gpus-per-node", type=int)
    plan.add_argument("--partition")
    plan.add_argument(
        "workload_command",
        nargs=argparse.REMAINDER,
        help="workload command after --",
    )

    validate = sub.add_parser(
        "validate",
        help="compare saved optimizer winners against promotion guardrails",
    )
    validate.add_argument("--baseline-optimization", required=True)
    validate.add_argument("--candidate-optimization", required=True)
    validate.add_argument("--optimization-store", default=".ago/optimizations")
    validate.add_argument("--validation-store", default=".ago/validations")
    validate.add_argument("--min-throughput-gain", type=float, default=0.0)
    validate.add_argument("--max-p99-regression", type=float)
    validate.add_argument("--max-energy-regression", type=float)
    validate.add_argument("--max-memory-regression", type=float)
    validate.add_argument("--no-rollback-on-reject", action="store_true")

    validations = sub.add_parser(
        "validations",
        help="inspect persisted promotion validation reports",
    )
    validations.add_argument("--store-dir", default=".ago/validations")
    validation_sub = validations.add_subparsers(
        dest="validations_action",
        required=True,
    )
    validation_sub.add_parser("list", help="list saved validation reports")
    validation_show = validation_sub.add_parser("show", help="show one report")
    validation_show.add_argument("identifier")

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

    if args.command == "optimize":
        try:
            candidates = generate_candidates(
                available_gpus=args.gpus,
                tensor_parallel=_csv_ints(args.tp),
                pipeline_parallel=_csv_ints(args.pp),
                data_parallel=_csv_ints(args.dp),
                batch_sizes=_csv_ints(args.batch),
                concurrencies=_csv_ints(args.concurrency),
                exact_world_size=args.exact_world_size,
                max_candidates=args.max_candidates,
            )
            if not candidates:
                raise ValueError("search space produced no valid candidates")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.dry_run:
            payload = [
                {
                    **candidate.to_dict(),
                    "world_size": candidate.world_size,
                    "label": candidate.label(),
                }
                for candidate in candidates
            ]
            if args.json:
                _print_json(payload)
            else:
                for candidate in candidates:
                    print(f"{candidate.label()}  world_size={candidate.world_size}")
            return 0

        command = list(args.benchmark_command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            print("error: optimize requires a benchmark command after --", file=sys.stderr)
            return 2

        constraints = OptimizationConstraints(
            max_p99_latency_ms=args.max_p99_ms,
            max_energy_joules=args.max_energy_joules,
            min_throughput_per_s=args.min_throughput,
            max_memory_pressure_pct=args.max_memory_pressure,
        )
        try:
            result = ConfigurationOptimizer(
                runner=CommandCanaryRunner(command=command, timeout_s=args.timeout),
                objective=OptimizationObjective(args.objective),
                constraints=constraints,
            ).optimize(candidates, baseline=candidates[0])
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        store = OptimizationStore(Path(args.store_dir))
        path = store.save(result)
        if args.json:
            payload = result.to_dict()
            payload["path"] = str(path)
            _print_json(payload)
        else:
            _plain_optimization(result)
            print(f"Saved: {path}")
        return 0 if result.best is not None else 3

    if args.command == "optimizations":
        store = OptimizationStore(Path(args.store_dir))
        if args.optimizations_action == "list":
            results = store.list()
            if args.json:
                _print_json([result.to_dict() for result in results])
            else:
                if not results:
                    print("No saved optimizations.")
                for result in results:
                    best = result.best.config.label() if result.best else "none"
                    print(
                        f"{result.id}  objective={result.objective.value}  "
                        f"trials={len(result.trials)}  best={best}"
                    )
            return 0

        result = store.load(args.identifier)
        _print_json(result.to_dict()) if args.json else _plain_optimization(result)
        return 0

    if args.command == "plan":
        command = list(args.workload_command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            print("error: plan requires a workload command after --", file=sys.stderr)
            return 2

        try:
            if args.optimization:
                optimized = OptimizationStore(Path(args.optimization_store)).load(
                    args.optimization
                )
                if optimized.best is None:
                    raise ValueError("saved optimization has no feasible best configuration")
                config = optimized.best.config
            else:
                config = ParallelismConfig(
                    tensor_parallel=args.tp,
                    pipeline_parallel=args.pp,
                    data_parallel=args.dp,
                    batch_size=args.batch,
                    concurrency=args.concurrency,
                )

            backend = SchedulerBackend(args.backend)
            policy = DeploymentPolicy(
                max_gpus=args.max_gpus,
                require_exact_gpu_count=args.require_exact_gpu_count,
                recommend_only=True,
            )
            deployment = DeploymentPlanner(policy).plan(
                config=config,
                backend=backend,
                command=command,
                available_gpus=args.available_gpus,
                node_selector=_parse_tags(args.node_selector),
            )

            if backend is SchedulerBackend.KUBERNETES:
                if not args.image:
                    raise ValueError("--image is required for Kubernetes plans")
                artifact = KubernetesRenderer(
                    image=args.image,
                    job_name=args.job_name,
                ).render(deployment)
            else:
                artifact = SlurmRenderer(
                    job_name=args.job_name,
                    nodes=args.nodes,
                    gpus_per_node=args.gpus_per_node,
                    partition=args.partition,
                ).render(deployment)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.json:
            _print_json({"plan": deployment.to_dict(), "artifact": artifact})
        else:
            for warning in deployment.warnings:
                print(f"warning: {warning}", file=sys.stderr)
            if isinstance(artifact, dict):
                _print_json(artifact)
            else:
                print(artifact, end="")
        return 0

    if args.command == "validate":
        try:
            optimization_store = OptimizationStore(Path(args.optimization_store))
            baseline_result = optimization_store.load(args.baseline_optimization)
            candidate_result = optimization_store.load(args.candidate_optimization)
            if baseline_result.best is None:
                raise ValueError("baseline optimization has no feasible best trial")
            if candidate_result.best is None:
                raise ValueError("candidate optimization has no feasible best trial")

            policy = ValidationPolicy(
                min_throughput_gain_pct=args.min_throughput_gain,
                max_p99_latency_regression_pct=args.max_p99_regression,
                max_energy_regression_pct=args.max_energy_regression,
                max_memory_pressure_regression_pct=args.max_memory_regression,
                rollback_on_reject=not args.no_rollback_on_reject,
            )
            report = validate_trials(
                baseline_result.best,
                candidate_result.best,
                policy,
            )
            path = ValidationStore(Path(args.validation_store)).save(report)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.json:
            payload = report.to_dict()
            payload["path"] = str(path)
            _print_json(payload)
        else:
            _plain_validation(report)
            print(f"Saved: {path}")
        return 0 if report.accepted else 3

    if args.command == "validations":
        store = ValidationStore(Path(args.store_dir))
        if args.validations_action == "list":
            reports = store.list()
            if args.json:
                _print_json([report.to_dict() for report in reports])
            else:
                if not reports:
                    print("No saved validations.")
                for report in reports:
                    print(
                        f"{report.id}  decision={report.decision.value}  "
                        f"{report.baseline_config.label()} -> "
                        f"{report.candidate_config.label()}"
                    )
            return 0

        try:
            report = store.load(args.identifier)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_json(report.to_dict()) if args.json else _plain_validation(report)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
