from __future__ import annotations

from statistics import mean

from adaptive_gpu_orchestrator.models import NCCLResult
from adaptive_gpu_orchestrator.optimizer import OptimizationResult
from adaptive_gpu_orchestrator.profiling import WorkloadProfile

from .models import CorrelationReport


def summarize_nccl(results: list[NCCLResult]) -> dict[str, float | int | None]:
    if not results:
        return {}

    bus = [float(item.busbw_gbps) for item in results]
    alg = [float(item.algbw_gbps) for item in results]
    largest = max(results, key=lambda item: item.size_bytes)

    return {
        "sample_count": len(results),
        "busbw_mean_gbps": mean(bus),
        "busbw_max_gbps": max(bus),
        "busbw_min_gbps": min(bus),
        "algbw_mean_gbps": mean(alg),
        "algbw_max_gbps": max(alg),
        "largest_message_bytes": largest.size_bytes,
        "largest_message_busbw_gbps": float(largest.busbw_gbps),
    }


def correlate_optimization(
    optimization: OptimizationResult,
    profile: WorkloadProfile,
    *,
    nccl_results: list[NCCLResult] | None = None,
) -> CorrelationReport:
    if optimization.best is None:
        raise ValueError("optimization has no feasible best trial")

    trial = optimization.best
    profile_summary = profile.summary()

    telemetry: dict[str, float | int | None] = {
        "duration_s": profile_summary.get("duration_s"),
        "sample_count": profile_summary.get("sample_count"),
        "gpu_count_max": profile_summary.get("gpu_count_max"),
        "gpu_util_mean_pct": profile_summary.get("gpu_util_mean_pct"),
        "gpu_util_std_pct": profile_summary.get("gpu_util_std_pct"),
        "memory_pressure_mean_pct": profile_summary.get("memory_pressure_mean_pct"),
        "temperature_max_c": profile_summary.get("temperature_max_c"),
        "power_draw_mean_w": profile_summary.get("power_draw_mean_w"),
    }

    duration = telemetry.get("duration_s")
    power = telemetry.get("power_draw_mean_w")
    if isinstance(duration, (int, float)) and isinstance(power, (int, float)):
        telemetry["estimated_energy_joules"] = float(duration) * float(power)

    throughput = trial.metrics.throughput_per_s
    estimated_energy = telemetry.get("estimated_energy_joules")
    if (
        isinstance(duration, (int, float))
        and duration > 0
        and isinstance(estimated_energy, (int, float))
        and throughput is not None
        and throughput > 0
    ):
        telemetry["energy_per_throughput_unit_j"] = float(estimated_energy) / (
            float(throughput) * float(duration)
        )

    nccl = summarize_nccl(nccl_results or [])
    adapter_metrics = dict(trial.metrics.extras)
    findings: list[str] = []

    util = telemetry.get("gpu_util_mean_pct")
    if isinstance(util, (int, float)) and util < 60:
        findings.append(
            f"Mean GPU utilization is {util:.1f}%; the selected configuration may be "
            "underfeeding accelerators."
        )

    imbalance = telemetry.get("gpu_util_std_pct")
    if isinstance(imbalance, (int, float)) and imbalance > 15:
        findings.append(
            f"GPU utilization standard deviation is {imbalance:.1f} percentage points; "
            "rank or pipeline imbalance is worth investigating."
        )

    memory = telemetry.get("memory_pressure_mean_pct")
    if isinstance(memory, (int, float)) and memory >= 90:
        findings.append(
            f"Mean memory pressure is {memory:.1f}%; memory headroom is limited."
        )

    temperature = telemetry.get("temperature_max_c")
    if isinstance(temperature, (int, float)) and temperature >= 85:
        findings.append(
            f"Peak temperature reached {temperature:.1f} C; thermal headroom is limited."
        )

    communication = adapter_metrics.get("communication_fraction_pct")
    if communication is not None and communication >= 25:
        if nccl:
            bandwidth = nccl.get("busbw_mean_gbps")
            findings.append(
                f"Communication accounts for {communication:.1f}% of step time; "
                f"NCCL mean bus bandwidth baseline is {bandwidth:.2f} GB/s."
            )
        else:
            findings.append(
                f"Communication accounts for {communication:.1f}% of step time; "
                "collect NCCL bandwidth data before increasing communication-heavy parallelism."
            )

    return CorrelationReport(
        optimization_id=optimization.id,
        profile_id=profile.id,
        config=trial.config,
        trial_metrics=trial.metrics,
        telemetry=telemetry,
        nccl=nccl,
        adapter_metrics=adapter_metrics,
        findings=findings,
    )
