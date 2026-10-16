from __future__ import annotations

from adaptive_gpu_orchestrator.controller import (
    DeploymentPlanner,
    DeploymentPolicy,
    SchedulerBackend,
)
from adaptive_gpu_orchestrator.correlation import CorrelationReport
from adaptive_gpu_orchestrator.optimizer import OptimizationResult
from adaptive_gpu_orchestrator.studies import StudyReport
from adaptive_gpu_orchestrator.timeseries import TimeSeriesReport
from adaptive_gpu_orchestrator.validation import diff_plans, validate_trials

from .models import (
    AutopilotDecision,
    AutopilotPolicy,
    AutopilotReport,
    RollbackPlan,
)


def _group_for_optimization(
    study: StudyReport,
    optimization_id: str,
):
    for group in study.groups:
        if optimization_id in group.optimization_ids:
            return group
    return None


def _rollback_plan(
    baseline_plan: dict,
    baseline_label: str,
    trigger_reasons: list[str],
) -> RollbackPlan:
    return RollbackPlan(
        target_config_label=baseline_label,
        baseline_plan=baseline_plan,
        trigger_reasons=tuple(trigger_reasons),
        steps=(
            "Keep scheduler mutation disabled in the autopilot process.",
            "Review the stored baseline deployment plan and candidate plan diff.",
            (
                f"Restore the baseline configuration {baseline_label} through an "
                "operator-controlled Kubernetes or Slurm workflow."
            ),
            "Re-run workload telemetry and guarded validation after rollback.",
            "Persist the rollback evidence before considering another promotion.",
        ),
    )


def decide_autopilot(
    *,
    baseline: OptimizationResult,
    candidate: OptimizationResult,
    backend: SchedulerBackend,
    command: list[str],
    available_gpus: int,
    policy: AutopilotPolicy | None = None,
    deployment_policy: DeploymentPolicy | None = None,
    node_selector: dict[str, str] | None = None,
    study: StudyReport | None = None,
    correlation: CorrelationReport | None = None,
    timeseries: TimeSeriesReport | None = None,
) -> AutopilotReport:
    active_policy = policy or AutopilotPolicy()

    if baseline.best is None:
        raise ValueError("baseline optimization has no feasible best trial")
    if candidate.best is None:
        raise ValueError("candidate optimization has no feasible best trial")
    if not command:
        raise ValueError("autopilot requires a workload command")
    if available_gpus <= 0:
        raise ValueError("available_gpus must be positive")

    active_deployment_policy = deployment_policy or DeploymentPolicy(
        max_gpus=available_gpus,
        require_exact_gpu_count=False,
        recommend_only=True,
    )
    if not active_deployment_policy.recommend_only:
        raise ValueError("v1.0 autopilot cannot use a mutating deployment policy")
    planner = DeploymentPlanner(active_deployment_policy)

    baseline_plan = planner.plan(
        config=baseline.best.config,
        backend=backend,
        command=command,
        available_gpus=available_gpus,
        node_selector=node_selector,
    )
    candidate_plan = planner.plan(
        config=candidate.best.config,
        backend=backend,
        command=command,
        available_gpus=available_gpus,
        node_selector=node_selector,
    )
    plan_diff = diff_plans(baseline_plan, candidate_plan)

    validation = validate_trials(
        baseline.best,
        candidate.best,
        active_policy.validation,
    )

    reject_reasons: list[str] = []
    hold_reasons: list[str] = []
    warnings: list[str] = []

    if not validation.accepted:
        reject_reasons.extend(validation.reasons)

    if active_policy.require_plan_change and not plan_diff.changed:
        hold_reasons.append(
            "candidate deployment plan is equivalent to the baseline plan"
        )

    if study is None:
        if active_policy.require_study:
            hold_reasons.append("required statistical study evidence is missing")
    else:
        group = _group_for_optimization(study, candidate.id)
        if group is None:
            message = (
                f"study {study.id} does not contain candidate optimization "
                f"{candidate.id}"
            )
            if active_policy.require_study:
                hold_reasons.append(message)
            else:
                warnings.append(message)
        else:
            replicate_count = len(group.optimization_ids)
            if replicate_count < active_policy.min_study_replicates:
                hold_reasons.append(
                    f"candidate study group has {replicate_count} replicates; "
                    f"{active_policy.min_study_replicates} required"
                )
            stability = group.config_stability_fraction
            if (
                stability is None
                or stability < active_policy.min_config_stability
            ):
                rendered = (
                    "unknown"
                    if stability is None
                    else f"{100.0 * stability:.1f}%"
                )
                hold_reasons.append(
                    f"candidate config-selection stability is {rendered}; "
                    f"{100.0 * active_policy.min_config_stability:.1f}% required"
                )

            if active_policy.require_pareto_candidate:
                frontier_ids = {
                    point.optimization_id
                    for point in study.pareto_frontier
                }
                if candidate.id not in frontier_ids:
                    hold_reasons.append(
                        "candidate is not on the study Pareto frontier"
                    )

    if correlation is None:
        if active_policy.require_correlation:
            hold_reasons.append("required correlation evidence is missing")
    else:
        if correlation.optimization_id != candidate.id:
            message = (
                f"correlation {correlation.id} references optimization "
                f"{correlation.optimization_id}, not candidate {candidate.id}"
            )
            if active_policy.require_correlation:
                hold_reasons.append(message)
            else:
                warnings.append(message)

        memory = correlation.telemetry.get("memory_pressure_mean_pct")
        if (
            isinstance(memory, (int, float))
            and memory >= active_policy.max_memory_pressure_pct
        ):
            hold_reasons.append(
                f"correlated mean memory pressure is {memory:.1f}%"
            )

        temperature = correlation.telemetry.get("temperature_max_c")
        if (
            isinstance(temperature, (int, float))
            and temperature >= active_policy.max_temperature_c
        ):
            hold_reasons.append(
                f"correlated peak temperature is {temperature:.1f} C"
            )

    if timeseries is None:
        if active_policy.require_timeseries:
            hold_reasons.append("required per-GPU time-series evidence is missing")
    else:
        if active_policy.hold_on_stragglers and timeseries.straggler_gpu_indices:
            hold_reasons.append(
                "per-GPU time-series analysis reports utilization stragglers: "
                + ", ".join(
                    str(index)
                    for index in timeseries.straggler_gpu_indices
                )
            )

        high_memory = [
            item.gpu_index
            for item in timeseries.gpu_summaries
            if item.memory_pressure_max_pct is not None
            and item.memory_pressure_max_pct
            >= active_policy.max_memory_pressure_pct
        ]
        if high_memory:
            hold_reasons.append(
                "per-GPU memory pressure exceeded the configured threshold on "
                + ", ".join(str(index) for index in high_memory)
            )

        hot = [
            item.gpu_index
            for item in timeseries.gpu_summaries
            if item.temperature_max_c is not None
            and item.temperature_max_c >= active_policy.max_temperature_c
        ]
        if hot:
            hold_reasons.append(
                "per-GPU temperature exceeded the configured threshold on "
                + ", ".join(str(index) for index in hot)
            )

    if reject_reasons:
        decision = AutopilotDecision.REJECT
        reasons = reject_reasons
    elif hold_reasons:
        decision = AutopilotDecision.HOLD
        reasons = hold_reasons
    else:
        decision = AutopilotDecision.PROMOTE
        reasons = [
            "candidate passed guarded validation and every configured evidence gate"
        ]

    rollback_triggers = reject_reasons + hold_reasons
    rollback = _rollback_plan(
        baseline_plan.to_dict(),
        baseline.best.config.label(),
        rollback_triggers,
    )

    return AutopilotReport(
        baseline_optimization_id=baseline.id,
        candidate_optimization_id=candidate.id,
        decision=decision,
        policy=active_policy,
        validation=validation,
        baseline_plan=baseline_plan.to_dict(),
        candidate_plan=candidate_plan.to_dict(),
        plan_diff=plan_diff.to_dict(),
        rollback=rollback,
        study_id=study.id if study is not None else None,
        correlation_id=correlation.id if correlation is not None else None,
        timeseries_id=timeseries.id if timeseries is not None else None,
        reasons=reasons,
        warnings=warnings,
    )
