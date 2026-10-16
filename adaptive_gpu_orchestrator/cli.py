from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adaptive_gpu_orchestrator.adapters import (
    AdaptedCommandCanaryRunner,
    PyTorchDistributedMetricsAdapter,
    VLLMMetricsAdapter,
)
from adaptive_gpu_orchestrator.autopilot import (
    AutopilotPolicy,
    AutopilotStore,
    decide_autopilot,
)
from adaptive_gpu_orchestrator.benchmarks import NCCLBenchmark, parse_nccl_output
from adaptive_gpu_orchestrator.campaigns import (
    CampaignNCCLBaseline,
    CampaignSpec,
    CampaignStore,
    capture_campaign_provenance,
    compare_campaigns,
    load_campaign_manifest,
    resume_campaign,
    run_campaign,
    save_campaign_manifest,
    write_campaign_export,
)
from adaptive_gpu_orchestrator.controller import (
    DeploymentPlanner,
    DeploymentPolicy,
    SchedulerBackend,
)
from adaptive_gpu_orchestrator.correlation import (
    CorrelationStore,
    correlate_optimization,
)
from adaptive_gpu_orchestrator.doctor import Doctor
from adaptive_gpu_orchestrator.evidence import (
    EvidenceBundleStore,
    build_evidence_bundle,
    capture_provenance,
    verify_bundle,
)
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
from adaptive_gpu_orchestrator.schedulers import (
    KubernetesJobSetRenderer,
    KubernetesRenderer,
    SlurmRenderer,
)
from adaptive_gpu_orchestrator.studies import StudyStore, analyze_study
from adaptive_gpu_orchestrator.telemetry import (
    NvidiaSMIProvider,
    NVMLProvider,
    NVMLUnavailable,
    parse_dcgm_dmon,
    parse_dcgm_exporter,
)
from adaptive_gpu_orchestrator.timeseries import (
    PhaseWindow,
    TimeSeriesStore,
    analyze_timeseries,
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


def _plain_correlation(report) -> None:
    print(
        f"Correlation: {report.id}  optimization={report.optimization_id}  "
        f"profile={report.profile_id}"
    )
    print(f"Config: {report.config.label()}")
    if report.telemetry:
        print("\nTelemetry:")
        for name, value in sorted(report.telemetry.items()):
            if value is not None:
                print(f"  - {name}: {value}")
    if report.nccl:
        print("\nNCCL:")
        for name, value in sorted(report.nccl.items()):
            if value is not None:
                print(f"  - {name}: {value}")
    if report.findings:
        print("\nFindings:")
        for finding in report.findings:
            print(f"  - {finding}")


def _plain_timeseries(report) -> None:
    print(
        f"Time-series: {report.id}  profile={report.profile_id}  "
        f"stragglers={report.straggler_gpu_indices or 'none'}"
    )
    if report.gpu_summaries:
        print("\nPer-GPU summary:")
        for item in report.gpu_summaries:
            util = (
                f"{item.gpu_util_mean_pct:.2f}%"
                if item.gpu_util_mean_pct is not None
                else "unknown"
            )
            gap = (
                f"{item.straggler_gap_pct_points:.2f}pp"
                if item.straggler_gap_pct_points is not None
                else "n/a"
            )
            print(
                f"  - GPU {item.gpu_index}: util={util} "
                f"straggler_gap={gap} samples={item.sample_count}"
            )
    if report.phases:
        print("\nPhases:")
        for phase in report.phases:
            print(
                f"  - {phase.name}: {phase.start_s:.3f}s-{phase.end_s:.3f}s "
                f"samples={phase.sample_count}"
            )
    if report.findings:
        print("\nFindings:")
        for finding in report.findings:
            print(f"  - {finding}")


def _plain_autopilot(report) -> None:
    print(
        f"Autopilot: {report.id}  decision={report.decision.value}  "
        f"promotable={report.promotable}"
    )
    print(
        f"Optimizations: {report.baseline_optimization_id} -> "
        f"{report.candidate_optimization_id}"
    )
    print(
        f"Plan changed: {report.plan_diff.get('changed', False)}  "
        f"rollback_target={report.rollback.target_config_label}"
    )

    evidence = []
    if report.study_id:
        evidence.append(f"study={report.study_id}")
    if report.correlation_id:
        evidence.append(f"correlation={report.correlation_id}")
    if report.timeseries_id:
        evidence.append(f"timeseries={report.timeseries_id}")
    if evidence:
        print("Evidence: " + "  ".join(evidence))

    if report.reasons:
        print("\nReasons:")
        for reason in report.reasons:
            print(f"  - {reason}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    print("\nRollback:")
    for step in report.rollback.steps:
        print(f"  - {step}")


def _plain_campaign(report) -> None:
    snapshot = report.provenance.cluster_snapshot
    gpus = snapshot.get("gpus", [])
    gpu_names = sorted(
        {
            str(item.get("name"))
            for item in gpus
            if item.get("name")
        }
    )
    print(
        f"Campaign: {report.id}  name={report.spec.name}  "
        f"group={report.spec.group}"
    )
    print(
        f"Replicates: {report.successful_replicates}/{report.spec.repeats}  "
        f"adapter={report.spec.adapter}  "
        f"objective={report.spec.objective.value}"
    )
    if report.manifest_sha256:
        print(f"Manifest SHA-256: {report.manifest_sha256}")
    print(
        f"Resume count: {report.resume_count}  "
        f"pending={list(report.pending_replicates) or 'none'}"
    )
    print(
        f"Hardware: GPUs={len(gpus)}  "
        f"models={','.join(gpu_names) if gpu_names else 'unknown'}  "
        f"driver={snapshot.get('driver_version') or 'unknown'}  "
        f"cuda={snapshot.get('cuda_version') or 'unknown'}"
    )
    if report.study_id:
        print(f"Study: {report.study_id}")
    if report.provenance.cuda_toolkit_version:
        print(f"CUDA toolkit: {report.provenance.cuda_toolkit_version}")
    if report.provenance.nccl_version:
        print(f"NCCL: {report.provenance.nccl_version}")
    if report.provenance.container_image:
        image = report.provenance.container_image
        digest = report.provenance.container_image_digest
        print(f"Container: {image}" + (f"@{digest}" if digest else ""))

    if report.nccl_baseline is not None:
        summary = report.nccl_baseline.summary
        print(
            "NCCL baseline: "
            f"samples={summary['sample_count']} "
            f"mean_busbw={summary['mean_busbw_gbps']} "
            f"peak_busbw={summary['peak_busbw_gbps']} Gbps"
        )

    print("\nRuns:")
    for run in report.runs:
        details = []
        if run.optimization_id:
            details.append(f"optimization={run.optimization_id}")
        if run.best_config:
            details.append(f"best={run.best_config}")
        if run.error:
            details.append(f"error={run.error}")
        suffix = "  " + "  ".join(details) if details else ""
        print(
            f"  - replicate={run.replicate} status={run.status}{suffix}"
        )

    if report.provenance.package_versions:
        print("\nSoftware:")
        for name, version in sorted(
            report.provenance.package_versions.items()
        ):
            print(f"  - {name}={version}")

    if report.findings:
        print("\nFindings:")
        for finding in report.findings:
            print(f"  - {finding}")


def _plain_bundle(bundle, verification=None) -> None:
    print(
        f"Evidence bundle: {bundle.id}  autopilot={bundle.autopilot_report_id}  "
        f"artifacts={len(bundle.artifacts)}"
    )
    print(f"SHA-256: {bundle.sha256}")

    provenance = bundle.provenance
    details = []
    if provenance.package_version:
        details.append(f"package={provenance.package_version}")
    if provenance.git_commit:
        details.append(f"git={provenance.git_commit[:12]}")
    details.append(f"python={provenance.python_version}")
    if details:
        print("Provenance: " + "  ".join(details))

    print("\nArtifacts:")
    for artifact in bundle.artifacts:
        print(
            f"  - {artifact.kind} {artifact.identifier} "
            f"sha256={artifact.sha256[:12]}..."
        )

    if verification is not None:
        print(
            f"\nIntegrity: valid={verification.valid}  "
            f"bundle_hash_matches={verification.bundle_hash_matches}"
        )
        for failure in verification.artifact_failures:
            print(f"  - failed artifact: {failure}")


def _plain_study(report) -> None:
    print(f"Study: {report.id}")
    for group in report.groups:
        stability = (
            f"{100.0 * group.config_stability_fraction:.1f}%"
            if group.config_stability_fraction is not None
            else "n/a"
        )
        print(
            f"  Group {group.name}: runs={len(group.optimization_ids)} "
            f"dominant={group.dominant_config or 'none'} stability={stability}"
        )
        for metric in (
            "throughput_per_s",
            "p99_latency_ms",
            "energy_joules",
        ):
            summary = group.metrics.get(metric)
            if summary is not None:
                interval = (
                    f" [{summary.ci95_low:.3f}, {summary.ci95_high:.3f}]"
                    if summary.ci95_low is not None
                    and summary.ci95_high is not None
                    else ""
                )
                print(
                    f"    {metric}: mean={summary.mean:.3f} n={summary.n}"
                    f"{interval}"
                )

    if report.comparisons:
        print("\nComparisons:")
        for item in report.comparisons:
            if item.metric not in {
                "throughput_per_s",
                "p99_latency_ms",
                "energy_joules",
            }:
                continue
            improvement = (
                f"{item.relative_improvement_pct:+.2f}%"
                if item.relative_improvement_pct is not None
                else "n/a"
            )
            effect = (
                f"{item.cohens_d:+.3f}"
                if item.cohens_d is not None
                else "n/a"
            )
            print(
                f"  - {item.candidate_group} vs {item.baseline_group} "
                f"{item.metric}: improvement={improvement} d={effect}"
            )

    if report.pareto_frontier:
        print("\nPareto frontier:")
        for point in report.pareto_frontier:
            print(
                f"  - {point.optimization_id} [{point.group}] "
                f"{point.config_label}"
            )

    if report.findings:
        print("\nFindings:")
        for finding in report.findings:
            print(f"  - {finding}")


def _study_group(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "study group must use LABEL=RUN_ID,RUN_ID,..."
        )
    name, raw_ids = value.split("=", 1)
    identifiers = tuple(
        item.strip()
        for item in raw_ids.split(",")
        if item.strip()
    )
    if not name.strip():
        raise argparse.ArgumentTypeError("study group label cannot be empty")
    if not identifiers:
        raise argparse.ArgumentTypeError(
            "study group requires at least one optimization id"
        )
    return name.strip(), identifiers


def _phase_window(value: str) -> PhaseWindow:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "phase must use NAME:START_SECONDS:END_SECONDS"
        )
    name, raw_start, raw_end = parts
    try:
        return PhaseWindow(
            name=name,
            start_s=float(raw_start),
            end_s=float(raw_end),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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


def _campaign_runner_factory(spec: CampaignSpec):
    command = list(spec.benchmark_command)

    def factory(replicate, campaign_env):
        if spec.adapter == "vllm":
            return AdaptedCommandCanaryRunner(
                adapter=VLLMMetricsAdapter(),
                command=command,
                timeout_s=spec.timeout_s,
                extra_env=campaign_env,
            )
        if spec.adapter == "pytorch-distributed":
            return AdaptedCommandCanaryRunner(
                adapter=PyTorchDistributedMetricsAdapter(),
                command=command,
                timeout_s=spec.timeout_s,
                extra_env=campaign_env,
            )
        return CommandCanaryRunner(
            command=command,
            timeout_s=spec.timeout_s,
            extra_env=campaign_env,
        )

    return factory


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
    optimize.add_argument(
        "--adapter",
        choices=("generic", "vllm", "pytorch-distributed"),
        default="generic",
        help="normalize native benchmark output before optimizer scoring",
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
    plan.add_argument(
        "--kubernetes-mode",
        choices=("job", "jobset"),
        default="job",
        help="render a single-pod Job or multi-node JobSet",
    )
    plan.add_argument("--nodes", type=int, default=1)
    plan.add_argument("--gpus-per-node", type=int)
    plan.add_argument("--queue-name")
    plan.add_argument("--master-port", type=int, default=29500)
    plan.add_argument("--service-account-name")
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

    correlate = sub.add_parser(
        "correlate",
        help="correlate an optimizer winner with telemetry and optional NCCL results",
    )
    correlate.add_argument("--optimization", required=True)
    correlate.add_argument("--profile", required=True)
    correlate.add_argument("--optimization-store", default=".ago/optimizations")
    correlate.add_argument("--experiment-store", default=".ago/experiments")
    correlate.add_argument("--nccl-output")
    correlate.add_argument("--store-dir", default=".ago/correlations")

    correlations = sub.add_parser(
        "correlations",
        help="inspect persisted telemetry correlation reports",
    )
    correlations.add_argument("--store-dir", default=".ago/correlations")
    correlation_sub = correlations.add_subparsers(
        dest="correlations_action",
        required=True,
    )
    correlation_sub.add_parser("list", help="list saved correlation reports")
    correlation_show = correlation_sub.add_parser("show", help="show one report")
    correlation_show.add_argument("identifier")

    timeseries = sub.add_parser(
        "timeseries",
        help="analyze and inspect per-GPU workload time series",
    )
    timeseries_sub = timeseries.add_subparsers(
        dest="timeseries_action",
        required=True,
    )

    timeseries_analyze = timeseries_sub.add_parser(
        "analyze",
        help="analyze a saved workload profile",
    )
    timeseries_analyze.add_argument("--profile", required=True)
    timeseries_analyze.add_argument(
        "--experiment-store",
        default=".ago/experiments",
    )
    timeseries_analyze.add_argument(
        "--phase",
        action="append",
        type=_phase_window,
        default=[],
        metavar="NAME:START:END",
    )
    timeseries_analyze.add_argument(
        "--straggler-gap",
        type=float,
        default=15.0,
        help="mean GPU-utilization gap in percentage points",
    )
    timeseries_analyze.add_argument("--dcgm-file")
    timeseries_analyze.add_argument(
        "--dcgm-format",
        choices=("exporter", "dmon"),
        default="exporter",
    )
    timeseries_analyze.add_argument("--dcgm-interval", type=float, default=1.0)
    timeseries_analyze.add_argument("--store-dir", default=".ago/timeseries")

    timeseries_list = timeseries_sub.add_parser(
        "list",
        help="list saved time-series reports",
    )
    timeseries_list.add_argument("--store-dir", default=".ago/timeseries")

    timeseries_show = timeseries_sub.add_parser(
        "show",
        help="show one saved time-series report",
    )
    timeseries_show.add_argument("identifier")
    timeseries_show.add_argument("--store-dir", default=".ago/timeseries")

    study = sub.add_parser(
        "study",
        help="analyze repeated optimization runs statistically",
    )
    study_sub = study.add_subparsers(
        dest="study_action",
        required=True,
    )
    study_analyze = study_sub.add_parser(
        "analyze",
        help="compare named groups of saved optimization runs",
    )
    study_analyze.add_argument(
        "--group",
        action="append",
        type=_study_group,
        required=True,
        metavar="LABEL=RUN_ID,RUN_ID",
    )
    study_analyze.add_argument(
        "--optimization-store",
        default=".ago/optimizations",
    )
    study_analyze.add_argument("--store-dir", default=".ago/studies")

    studies = sub.add_parser(
        "studies",
        help="inspect persisted statistical study reports",
    )
    studies_sub = studies.add_subparsers(
        dest="studies_action",
        required=True,
    )
    studies_list = studies_sub.add_parser("list", help="list saved studies")
    studies_list.add_argument("--store-dir", default=".ago/studies")
    studies_show = studies_sub.add_parser("show", help="show one study")
    studies_show.add_argument("identifier")
    studies_show.add_argument("--store-dir", default=".ago/studies")

    autopilot = sub.add_parser(
        "autopilot",
        help="make an auditable recommend-only promote/hold/reject decision",
    )
    autopilot_sub = autopilot.add_subparsers(
        dest="autopilot_action",
        required=True,
    )

    autopilot_decide = autopilot_sub.add_parser(
        "decide",
        help="evaluate a candidate using validation, evidence, and deployment policy",
    )
    autopilot_decide.add_argument("--baseline-optimization", required=True)
    autopilot_decide.add_argument("--candidate-optimization", required=True)
    autopilot_decide.add_argument(
        "--backend",
        choices=[item.value for item in SchedulerBackend],
        required=True,
    )
    autopilot_decide.add_argument("--available-gpus", type=int, required=True)
    autopilot_decide.add_argument(
        "--optimization-store",
        default=".ago/optimizations",
    )
    autopilot_decide.add_argument("--study")
    autopilot_decide.add_argument("--study-store", default=".ago/studies")
    autopilot_decide.add_argument("--correlation")
    autopilot_decide.add_argument(
        "--correlation-store",
        default=".ago/correlations",
    )
    autopilot_decide.add_argument("--timeseries")
    autopilot_decide.add_argument(
        "--timeseries-store",
        default=".ago/timeseries",
    )
    autopilot_decide.add_argument("--store-dir", default=".ago/autopilot")
    autopilot_decide.add_argument("--min-throughput-gain", type=float, default=0.0)
    autopilot_decide.add_argument("--max-p99-regression", type=float)
    autopilot_decide.add_argument("--max-energy-regression", type=float)
    autopilot_decide.add_argument("--max-memory-regression", type=float)
    autopilot_decide.add_argument("--require-study", action="store_true")
    autopilot_decide.add_argument("--require-correlation", action="store_true")
    autopilot_decide.add_argument("--require-timeseries", action="store_true")
    autopilot_decide.add_argument(
        "--require-pareto-candidate",
        action="store_true",
    )
    autopilot_decide.add_argument("--min-study-replicates", type=int, default=3)
    autopilot_decide.add_argument(
        "--min-config-stability",
        type=float,
        default=0.67,
    )
    autopilot_decide.add_argument("--allow-stragglers", action="store_true")
    autopilot_decide.add_argument(
        "--max-memory-pressure",
        type=float,
        default=95.0,
    )
    autopilot_decide.add_argument(
        "--max-temperature",
        type=float,
        default=85.0,
    )
    autopilot_decide.add_argument(
        "--allow-equivalent-plan",
        action="store_true",
    )
    autopilot_decide.add_argument("--max-gpus", type=int)
    autopilot_decide.add_argument(
        "--require-exact-gpu-count",
        action="store_true",
    )
    autopilot_decide.add_argument(
        "--node-selector",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    autopilot_decide.add_argument(
        "workload_command",
        nargs=argparse.REMAINDER,
        help="workload command after --",
    )

    autopilot_list = autopilot_sub.add_parser(
        "list",
        help="list saved autopilot decisions",
    )
    autopilot_list.add_argument("--store-dir", default=".ago/autopilot")

    autopilot_show = autopilot_sub.add_parser(
        "show",
        help="show one saved autopilot decision",
    )
    autopilot_show.add_argument("identifier")
    autopilot_show.add_argument("--store-dir", default=".ago/autopilot")

    campaign = sub.add_parser(
        "campaign",
        help="run, resume and compare reproducible optimization campaigns",
    )
    campaign_sub = campaign.add_subparsers(
        dest="campaign_action",
        required=True,
    )

    campaign_run = campaign_sub.add_parser(
        "run",
        help="run repeated optimizer replicates with captured provenance",
    )
    campaign_run.add_argument("--manifest")
    campaign_run.add_argument("--name")
    campaign_run.add_argument("--group")
    campaign_run.add_argument("--repeats", type=int, default=3)
    campaign_run.add_argument("--gpus", type=int)
    campaign_run.add_argument("--tp", default="1,2,4,8")
    campaign_run.add_argument("--pp", default="1,2")
    campaign_run.add_argument("--dp", default="1")
    campaign_run.add_argument("--batch", default="1,2,4,8")
    campaign_run.add_argument("--concurrency", default="1,8")
    campaign_run.add_argument(
        "--objective",
        choices=[item.value for item in OptimizationObjective],
        default=OptimizationObjective.THROUGHPUT.value,
    )
    campaign_run.add_argument(
        "--adapter",
        choices=("generic", "vllm", "pytorch-distributed"),
        default="generic",
    )
    campaign_run.add_argument("--exact-world-size", action="store_true")
    campaign_run.add_argument("--max-candidates", type=int, default=64)
    campaign_run.add_argument("--timeout", type=float, default=120.0)
    campaign_run.add_argument("--max-p99-ms", type=float)
    campaign_run.add_argument("--max-energy-joules", type=float)
    campaign_run.add_argument("--min-throughput", type=float)
    campaign_run.add_argument("--max-memory-pressure", type=float)
    campaign_run.add_argument(
        "--randomize-candidates",
        action="store_true",
    )
    campaign_run.add_argument("--random-seed", type=int)
    campaign_run.add_argument(
        "--optimization-store",
        default=".ago/optimizations",
    )
    campaign_run.add_argument("--study-store", default=".ago/studies")
    campaign_run.add_argument("--store-dir", default=".ago/campaigns")
    campaign_run.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    campaign_run.add_argument("--cuda-toolkit-version")
    campaign_run.add_argument("--nccl-version")
    campaign_run.add_argument("--container-image")
    campaign_run.add_argument("--container-image-digest")
    nccl_source = campaign_run.add_mutually_exclusive_group()
    nccl_source.add_argument(
        "--nccl-baseline",
        action="store_true",
        help="run nccl-tests once before campaign replicates",
    )
    nccl_source.add_argument(
        "--nccl-output",
        help="ingest saved nccl-tests output instead of running a baseline",
    )
    campaign_run.add_argument("--nccl-binary", default="all_reduce_perf")
    campaign_run.add_argument("--nccl-begin", default="8M")
    campaign_run.add_argument("--nccl-end", default="1G")
    campaign_run.add_argument("--nccl-factor", type=int, default=2)
    campaign_run.add_argument("--nccl-gpus", type=int)
    campaign_run.add_argument(
        "benchmark_command",
        nargs=argparse.REMAINDER,
        help="benchmark command after --; omit when --manifest is used",
    )

    campaign_resume = campaign_sub.add_parser(
        "resume",
        help="rerun only incomplete replicates in a saved campaign",
    )
    campaign_resume.add_argument("identifier")
    campaign_resume.add_argument(
        "--optimization-store",
        default=".ago/optimizations",
    )
    campaign_resume.add_argument("--study-store", default=".ago/studies")
    campaign_resume.add_argument("--store-dir", default=".ago/campaigns")
    campaign_resume.add_argument(
        "--allow-provenance-change",
        action="store_true",
    )
    campaign_resume.add_argument("--cuda-toolkit-version")
    campaign_resume.add_argument("--nccl-version")
    campaign_resume.add_argument("--container-image")
    campaign_resume.add_argument("--container-image-digest")

    campaign_export = campaign_sub.add_parser(
        "export",
        help="export every optimizer trial as analysis-ready CSV or JSONL",
    )
    campaign_export.add_argument("identifier")
    campaign_export.add_argument("--output", required=True)
    campaign_export.add_argument(
        "--format",
        choices=("csv", "jsonl"),
        default="csv",
    )
    campaign_export.add_argument("--store-dir", default=".ago/campaigns")
    campaign_export.add_argument(
        "--optimization-store",
        default=".ago/optimizations",
    )

    campaign_manifest = campaign_sub.add_parser(
        "manifest",
        help="export a hash-verified manifest from a saved campaign",
    )
    campaign_manifest.add_argument("identifier")
    campaign_manifest.add_argument("--output", required=True)
    campaign_manifest.add_argument("--store-dir", default=".ago/campaigns")

    campaign_list = campaign_sub.add_parser(
        "list",
        help="list saved campaigns",
    )
    campaign_list.add_argument("--store-dir", default=".ago/campaigns")

    campaign_show = campaign_sub.add_parser(
        "show",
        help="show one saved campaign",
    )
    campaign_show.add_argument("identifier")
    campaign_show.add_argument("--store-dir", default=".ago/campaigns")

    campaign_compare = campaign_sub.add_parser(
        "compare",
        help="build a statistical study from completed campaign replicates",
    )
    campaign_compare.add_argument("identifiers", nargs="+")
    campaign_compare.add_argument("--store-dir", default=".ago/campaigns")
    campaign_compare.add_argument(
        "--optimization-store",
        default=".ago/optimizations",
    )
    campaign_compare.add_argument("--study-store", default=".ago/studies")

    bundle = sub.add_parser(
        "bundle",
        help="create and verify portable evidence bundles",
    )
    bundle_sub = bundle.add_subparsers(
        dest="bundle_action",
        required=True,
    )

    bundle_create = bundle_sub.add_parser(
        "create",
        help="bundle an autopilot decision with every referenced evidence artifact",
    )
    bundle_create.add_argument("--autopilot", required=True)
    bundle_create.add_argument("--autopilot-store", default=".ago/autopilot")
    bundle_create.add_argument(
        "--optimization-store",
        default=".ago/optimizations",
    )
    bundle_create.add_argument("--study-store", default=".ago/studies")
    bundle_create.add_argument(
        "--correlation-store",
        default=".ago/correlations",
    )
    bundle_create.add_argument(
        "--timeseries-store",
        default=".ago/timeseries",
    )
    bundle_create.add_argument("--store-dir", default=".ago/bundles")
    bundle_create.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="attach reproducibility metadata; may be repeated",
    )

    bundle_list = bundle_sub.add_parser(
        "list",
        help="list saved evidence bundles",
    )
    bundle_list.add_argument("--store-dir", default=".ago/bundles")

    bundle_show = bundle_sub.add_parser(
        "show",
        help="show and verify one evidence bundle",
    )
    bundle_show.add_argument("identifier")
    bundle_show.add_argument("--store-dir", default=".ago/bundles")

    bundle_verify = bundle_sub.add_parser(
        "verify",
        help="verify all artifact fingerprints and the bundle fingerprint",
    )
    bundle_verify.add_argument("identifier")
    bundle_verify.add_argument("--store-dir", default=".ago/bundles")

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
            if args.adapter == "vllm":
                runner = AdaptedCommandCanaryRunner(
                    adapter=VLLMMetricsAdapter(),
                    command=command,
                    timeout_s=args.timeout,
                )
            elif args.adapter == "pytorch-distributed":
                runner = AdaptedCommandCanaryRunner(
                    adapter=PyTorchDistributedMetricsAdapter(),
                    command=command,
                    timeout_s=args.timeout,
                )
            else:
                runner = CommandCanaryRunner(
                    command=command,
                    timeout_s=args.timeout,
                )

            result = ConfigurationOptimizer(
                runner=runner,
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

                if args.kubernetes_mode == "jobset":
                    if args.nodes <= 0:
                        raise ValueError("--nodes must be positive")
                    if args.gpus_per_node is None:
                        if config.world_size % args.nodes != 0:
                            raise ValueError(
                                "world size must be divisible by --nodes when "
                                "--gpus-per-node is omitted"
                            )
                        gpus_per_node = config.world_size // args.nodes
                    else:
                        gpus_per_node = args.gpus_per_node

                    artifact = KubernetesJobSetRenderer(
                        image=args.image,
                        job_name=args.job_name,
                        nodes=args.nodes,
                        gpus_per_node=gpus_per_node,
                        queue_name=args.queue_name,
                        master_port=args.master_port,
                        service_account_name=args.service_account_name,
                    ).render(deployment)
                else:
                    if args.nodes != 1:
                        raise ValueError(
                            "single-pod Kubernetes Job requires --nodes 1; "
                            "use --kubernetes-mode jobset for multi-node plans"
                        )
                    if (
                        args.gpus_per_node is not None
                        and args.gpus_per_node != config.world_size
                    ):
                        raise ValueError(
                            "single-pod Kubernetes Job requires --gpus-per-node "
                            "to match the configuration world size"
                        )
                    artifact = KubernetesRenderer(
                        image=args.image,
                        job_name=args.job_name,
                        queue_name=args.queue_name,
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

    if args.command == "correlate":
        try:
            optimization = OptimizationStore(Path(args.optimization_store)).load(
                args.optimization
            )
            profile = ExperimentStore(Path(args.experiment_store)).load(args.profile)

            nccl_results = None
            if args.nccl_output:
                with open(args.nccl_output, "r", encoding="utf-8") as handle:
                    nccl_results = parse_nccl_output(handle.read())

            report = correlate_optimization(
                optimization,
                profile,
                nccl_results=nccl_results,
            )
            path = CorrelationStore(Path(args.store_dir)).save(report)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.json:
            payload = report.to_dict()
            payload["path"] = str(path)
            _print_json(payload)
        else:
            _plain_correlation(report)
            print(f"Saved: {path}")
        return 0

    if args.command == "correlations":
        store = CorrelationStore(Path(args.store_dir))
        if args.correlations_action == "list":
            reports = store.list()
            if args.json:
                _print_json([report.to_dict() for report in reports])
            else:
                if not reports:
                    print("No saved correlations.")
                for report in reports:
                    print(
                        f"{report.id}  optimization={report.optimization_id}  "
                        f"profile={report.profile_id}  config={report.config.label()}"
                    )
            return 0

        try:
            report = store.load(args.identifier)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_json(report.to_dict()) if args.json else _plain_correlation(report)
        return 0

    if args.command == "timeseries":
        store = TimeSeriesStore(Path(args.store_dir))

        if args.timeseries_action == "analyze":
            try:
                profile = ExperimentStore(Path(args.experiment_store)).load(
                    args.profile
                )
                dcgm_frames = []
                if args.dcgm_file:
                    with open(args.dcgm_file, "r", encoding="utf-8") as handle:
                        dcgm_text = handle.read()
                    if args.dcgm_format == "dmon":
                        dcgm_frames = parse_dcgm_dmon(
                            dcgm_text,
                            interval_s=args.dcgm_interval,
                        )
                    else:
                        dcgm_frames = parse_dcgm_exporter(
                            dcgm_text,
                            interval_s=args.dcgm_interval,
                        )

                report = analyze_timeseries(
                    profile,
                    phases=args.phase,
                    dcgm_frames=dcgm_frames,
                    straggler_gap_pct_points=args.straggler_gap,
                )
                path = store.save(report)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                payload = report.to_dict()
                payload["path"] = str(path)
                _print_json(payload)
            else:
                _plain_timeseries(report)
                print(f"Saved: {path}")
            return 0

        if args.timeseries_action == "list":
            reports = store.list()
            if args.json:
                _print_json([report.to_dict() for report in reports])
            else:
                if not reports:
                    print("No saved time-series reports.")
                for report in reports:
                    print(
                        f"{report.id}  profile={report.profile_id}  "
                        f"stragglers={report.straggler_gpu_indices or 'none'}"
                    )
            return 0

        try:
            report = store.load(args.identifier)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_json(report.to_dict()) if args.json else _plain_timeseries(report)
        return 0

    if args.command == "study":
        optimization_store = OptimizationStore(
            Path(args.optimization_store)
        )
        groups = {}
        try:
            for name, identifiers in args.group:
                if name in groups:
                    raise ValueError(
                        f"duplicate study group label: {name}"
                    )
                groups[name] = [
                    optimization_store.load(identifier)
                    for identifier in identifiers
                ]

            report = analyze_study(groups)
            path = StudyStore(Path(args.store_dir)).save(report)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.json:
            payload = report.to_dict()
            payload["path"] = str(path)
            _print_json(payload)
        else:
            _plain_study(report)
            print(f"Saved: {path}")
        return 0

    if args.command == "studies":
        store = StudyStore(Path(args.store_dir))
        if args.studies_action == "list":
            reports = store.list()
            if args.json:
                _print_json([report.to_dict() for report in reports])
            else:
                if not reports:
                    print("No saved studies.")
                for report in reports:
                    groups = ",".join(
                        group.name
                        for group in report.groups
                    )
                    print(f"{report.id}  groups={groups}")
            return 0

        try:
            report = store.load(args.identifier)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_json(report.to_dict()) if args.json else _plain_study(report)
        return 0

    if args.command == "autopilot":
        store = AutopilotStore(Path(args.store_dir))

        if args.autopilot_action == "decide":
            command = list(args.workload_command)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                print(
                    "error: autopilot decide requires a workload command after --",
                    file=sys.stderr,
                )
                return 2

            try:
                optimization_store = OptimizationStore(
                    Path(args.optimization_store)
                )
                baseline = optimization_store.load(args.baseline_optimization)
                candidate = optimization_store.load(args.candidate_optimization)

                study = (
                    StudyStore(Path(args.study_store)).load(args.study)
                    if args.study
                    else None
                )
                correlation = (
                    CorrelationStore(Path(args.correlation_store)).load(
                        args.correlation
                    )
                    if args.correlation
                    else None
                )
                timeseries = (
                    TimeSeriesStore(Path(args.timeseries_store)).load(
                        args.timeseries
                    )
                    if args.timeseries
                    else None
                )

                validation_policy = ValidationPolicy(
                    min_throughput_gain_pct=args.min_throughput_gain,
                    max_p99_latency_regression_pct=args.max_p99_regression,
                    max_energy_regression_pct=args.max_energy_regression,
                    max_memory_pressure_regression_pct=args.max_memory_regression,
                    rollback_on_reject=True,
                )
                autopilot_policy = AutopilotPolicy(
                    validation=validation_policy,
                    min_study_replicates=args.min_study_replicates,
                    min_config_stability=args.min_config_stability,
                    require_study=args.require_study,
                    require_correlation=args.require_correlation,
                    require_timeseries=args.require_timeseries,
                    require_pareto_candidate=args.require_pareto_candidate,
                    hold_on_stragglers=not args.allow_stragglers,
                    max_memory_pressure_pct=args.max_memory_pressure,
                    max_temperature_c=args.max_temperature,
                    require_plan_change=not args.allow_equivalent_plan,
                    recommend_only=True,
                )
                deployment_policy = DeploymentPolicy(
                    max_gpus=args.max_gpus,
                    require_exact_gpu_count=args.require_exact_gpu_count,
                    recommend_only=True,
                )

                report = decide_autopilot(
                    baseline=baseline,
                    candidate=candidate,
                    backend=SchedulerBackend(args.backend),
                    command=command,
                    available_gpus=args.available_gpus,
                    policy=autopilot_policy,
                    deployment_policy=deployment_policy,
                    node_selector=_parse_tags(args.node_selector),
                    study=study,
                    correlation=correlation,
                    timeseries=timeseries,
                )
                path = store.save(report)
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                payload = report.to_dict()
                payload["path"] = str(path)
                _print_json(payload)
            else:
                _plain_autopilot(report)
                print(f"Saved: {path}")

            if report.decision.value == "promote":
                return 0
            if report.decision.value == "reject":
                return 3
            return 4

        if args.autopilot_action == "list":
            reports = store.list()
            if args.json:
                _print_json([report.to_dict() for report in reports])
            else:
                if not reports:
                    print("No saved autopilot decisions.")
                for report in reports:
                    print(
                        f"{report.id}  decision={report.decision.value}  "
                        f"{report.baseline_optimization_id} -> "
                        f"{report.candidate_optimization_id}"
                    )
            return 0

        try:
            report = store.load(args.identifier)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_json(report.to_dict()) if args.json else _plain_autopilot(report)
        return 0

    if args.command == "campaign":
        store = CampaignStore(Path(args.store_dir))

        if args.campaign_action == "run":
            try:
                if args.manifest:
                    if args.benchmark_command:
                        raise ValueError(
                            "benchmark command cannot be supplied with --manifest"
                        )
                    if args.tag:
                        raise ValueError(
                            "--tag cannot override a hash-verified manifest"
                        )
                    manifest = load_campaign_manifest(args.manifest)
                    spec = manifest.spec
                else:
                    command = list(args.benchmark_command)
                    if command and command[0] == "--":
                        command = command[1:]
                    if not command:
                        raise ValueError(
                            "campaign run requires a benchmark command after --"
                        )
                    if not args.name or not args.group or args.gpus is None:
                        raise ValueError(
                            "--name, --group and --gpus are required "
                            "when --manifest is not used"
                        )

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
                        raise ValueError(
                            "campaign search space produced no valid candidates"
                        )

                    tags = _parse_tags(args.tag)
                    constraints = OptimizationConstraints(
                        max_p99_latency_ms=args.max_p99_ms,
                        max_energy_joules=args.max_energy_joules,
                        min_throughput_per_s=args.min_throughput,
                        max_memory_pressure_pct=args.max_memory_pressure,
                    )
                    spec = CampaignSpec(
                        name=args.name,
                        group=args.group,
                        repeats=args.repeats,
                        available_gpus=args.gpus,
                        candidates=tuple(candidates),
                        objective=OptimizationObjective(args.objective),
                        constraints=constraints,
                        adapter=args.adapter,
                        benchmark_command=tuple(command),
                        timeout_s=args.timeout,
                        randomize_candidates=args.randomize_candidates,
                        random_seed=args.random_seed,
                        tags=tags,
                    )

                provenance = capture_campaign_provenance(
                    tags=spec.tags,
                    cuda_toolkit_version=args.cuda_toolkit_version,
                    nccl_version=args.nccl_version,
                    container_image=args.container_image,
                    container_image_digest=args.container_image_digest,
                )

                nccl_baseline = None
                nccl_gpus = args.nccl_gpus or spec.available_gpus
                if args.nccl_output:
                    with open(args.nccl_output, "r", encoding="utf-8") as handle:
                        nccl_results = parse_nccl_output(handle.read())
                    if not nccl_results:
                        raise ValueError(
                            "saved NCCL output contained no benchmark rows"
                        )
                    nccl_baseline = CampaignNCCLBaseline(
                        operation="all_reduce",
                        source="saved-output",
                        binary=args.nccl_binary,
                        begin=args.nccl_begin,
                        end=args.nccl_end,
                        factor=args.nccl_factor,
                        gpus=nccl_gpus,
                        results=tuple(nccl_results),
                    )
                elif args.nccl_baseline:
                    nccl_results = NCCLBenchmark(
                        binary=args.nccl_binary
                    ).run(
                        begin=args.nccl_begin,
                        end=args.nccl_end,
                        factor=args.nccl_factor,
                        gpus=nccl_gpus,
                    )
                    if not nccl_results:
                        raise ValueError(
                            "NCCL baseline produced no benchmark rows"
                        )
                    nccl_baseline = CampaignNCCLBaseline(
                        operation="all_reduce",
                        source="executed",
                        binary=args.nccl_binary,
                        begin=args.nccl_begin,
                        end=args.nccl_end,
                        factor=args.nccl_factor,
                        gpus=nccl_gpus,
                        results=tuple(nccl_results),
                    )

                report = run_campaign(
                    spec=spec,
                    provenance=provenance,
                    runner_factory=_campaign_runner_factory(spec),
                    optimization_store=OptimizationStore(
                        Path(args.optimization_store)
                    ),
                    study_store=StudyStore(Path(args.study_store)),
                    nccl_baseline=nccl_baseline,
                )
                path = store.save(report)
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                payload = report.to_dict()
                payload["path"] = str(path)
                _print_json(payload)
            else:
                _plain_campaign(report)
                print(f"Saved: {path}")
            return 0 if report.successful_replicates else 3

        if args.campaign_action == "resume":
            try:
                report = store.load(args.identifier)
                provenance = capture_campaign_provenance(
                    tags=report.spec.tags,
                    cuda_toolkit_version=args.cuda_toolkit_version,
                    nccl_version=args.nccl_version,
                    container_image=args.container_image,
                    container_image_digest=args.container_image_digest,
                )
                report = resume_campaign(
                    report,
                    runner_factory=_campaign_runner_factory(report.spec),
                    optimization_store=OptimizationStore(
                        Path(args.optimization_store)
                    ),
                    study_store=StudyStore(Path(args.study_store)),
                    current_provenance=provenance,
                    allow_provenance_change=args.allow_provenance_change,
                )
                path = store.save(report)
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                payload = report.to_dict()
                payload["path"] = str(path)
                _print_json(payload)
            else:
                _plain_campaign(report)
                print(f"Saved: {path}")
            return 0 if not report.pending_replicates else 3

        if args.campaign_action == "export":
            try:
                report = store.load(args.identifier)
                path = write_campaign_export(
                    report,
                    optimization_store=OptimizationStore(
                        Path(args.optimization_store)
                    ),
                    output=args.output,
                    format=args.format,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                _print_json(
                    {
                        "campaign_id": report.id,
                        "format": args.format,
                        "path": str(path),
                    }
                )
            else:
                print(
                    f"Exported campaign {report.id} to {path} "
                    f"({args.format})"
                )
            return 0

        if args.campaign_action == "manifest":
            try:
                report = store.load(args.identifier)
                manifest = save_campaign_manifest(
                    report.spec,
                    args.output,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                _print_json(
                    {
                        "path": str(args.output),
                        "manifest": manifest.to_dict(),
                    }
                )
            else:
                print(
                    f"Manifest: {args.output}  sha256={manifest.sha256}"
                )
            return 0

        if args.campaign_action == "list":
            reports = store.list()
            if args.json:
                _print_json([report.to_dict() for report in reports])
            else:
                if not reports:
                    print("No saved campaigns.")
                for report in reports:
                    print(
                        f"{report.id}  name={report.spec.name}  "
                        f"group={report.spec.group}  "
                        f"replicates={report.successful_replicates}/"
                        f"{report.spec.repeats}  "
                        f"pending={len(report.pending_replicates)}"
                    )
            return 0

        if args.campaign_action == "compare":
            try:
                campaigns = [
                    store.load(identifier)
                    for identifier in args.identifiers
                ]
                study = compare_campaigns(
                    campaigns,
                    optimization_store=OptimizationStore(
                        Path(args.optimization_store)
                    ),
                    study_store=StudyStore(Path(args.study_store)),
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                _print_json(study.to_dict())
            else:
                _plain_study(study)
                print(f"Saved study: {args.study_store}/{study.id}.json")
            return 0

        try:
            report = store.load(args.identifier)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_json(report.to_dict()) if args.json else _plain_campaign(report)
        return 0

    if args.command == "bundle":
        store = EvidenceBundleStore(Path(args.store_dir))

        if args.bundle_action == "create":
            try:
                autopilot = AutopilotStore(
                    Path(args.autopilot_store)
                ).load(args.autopilot)
                optimization_store = OptimizationStore(
                    Path(args.optimization_store)
                )
                baseline = optimization_store.load(
                    autopilot.baseline_optimization_id
                )
                candidate = optimization_store.load(
                    autopilot.candidate_optimization_id
                )

                study = (
                    StudyStore(Path(args.study_store)).load(
                        autopilot.study_id
                    )
                    if autopilot.study_id
                    else None
                )
                correlation = (
                    CorrelationStore(Path(args.correlation_store)).load(
                        autopilot.correlation_id
                    )
                    if autopilot.correlation_id
                    else None
                )
                timeseries = (
                    TimeSeriesStore(Path(args.timeseries_store)).load(
                        autopilot.timeseries_id
                    )
                    if autopilot.timeseries_id
                    else None
                )

                provenance = capture_provenance(
                    tags=_parse_tags(args.tag)
                )
                bundle = build_evidence_bundle(
                    autopilot=autopilot,
                    baseline=baseline,
                    candidate=candidate,
                    provenance=provenance,
                    study=study,
                    correlation=correlation,
                    timeseries=timeseries,
                )
                path = store.save(bundle)
                verification = verify_bundle(bundle)
            except (FileNotFoundError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            if args.json:
                payload = bundle.to_dict()
                payload["path"] = str(path)
                payload["verification"] = verification.to_dict()
                _print_json(payload)
            else:
                _plain_bundle(bundle, verification)
                print(f"Saved: {path}")
            return 0

        if args.bundle_action == "list":
            bundles = store.list()
            if args.json:
                _print_json([bundle.to_dict() for bundle in bundles])
            else:
                if not bundles:
                    print("No saved evidence bundles.")
                for bundle in bundles:
                    print(
                        f"{bundle.id}  autopilot={bundle.autopilot_report_id}  "
                        f"artifacts={len(bundle.artifacts)}  "
                        f"sha256={bundle.sha256[:12]}..."
                    )
            return 0

        try:
            bundle = store.load(args.identifier)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        verification = verify_bundle(bundle)
        if args.json:
            _print_json(
                {
                    "bundle": bundle.to_dict(),
                    "verification": verification.to_dict(),
                }
            )
        else:
            _plain_bundle(bundle, verification)

        if args.bundle_action == "verify":
            return 0 if verification.valid else 5
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
