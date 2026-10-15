from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

from adaptive_gpu_orchestrator.profiling import WorkloadProfile
from adaptive_gpu_orchestrator.telemetry import DCGMFrame, summarize_dcgm

from .models import (
    GPUTimeSeriesSummary,
    PhaseSummary,
    PhaseWindow,
    TimeSeriesReport,
)


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _phase_summary(
    profile: WorkloadProfile,
    phase: PhaseWindow,
) -> PhaseSummary:
    samples = [
        sample
        for sample in profile.samples
        if phase.start_s <= sample.offset_s < phase.end_s
    ]

    util = [
        float(sample.gpu_util_mean_pct)
        for sample in samples
        if sample.gpu_util_mean_pct is not None
    ]
    memory = [
        float(sample.memory_pressure_mean_pct)
        for sample in samples
        if sample.memory_pressure_mean_pct is not None
    ]
    power = [
        float(sample.power_draw_total_w)
        for sample in samples
        if sample.power_draw_total_w is not None
    ]

    by_gpu: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        for device in sample.devices:
            if device.gpu_util_pct is not None:
                by_gpu[device.index].append(float(device.gpu_util_pct))

    return PhaseSummary(
        name=phase.name,
        start_s=phase.start_s,
        end_s=phase.end_s,
        sample_count=len(samples),
        gpu_util_mean_pct=_avg(util),
        memory_pressure_mean_pct=_avg(memory),
        power_draw_mean_w=_avg(power),
        gpu_util_by_index={
            index: mean(values)
            for index, values in sorted(by_gpu.items())
            if values
        },
    )


def analyze_timeseries(
    profile: WorkloadProfile,
    *,
    phases: list[PhaseWindow] | None = None,
    dcgm_frames: list[DCGMFrame] | None = None,
    straggler_gap_pct_points: float = 15.0,
) -> TimeSeriesReport:
    if straggler_gap_pct_points < 0:
        raise ValueError("straggler gap must be non-negative")

    by_gpu: dict[int, list] = defaultdict(list)
    for sample in profile.samples:
        for device in sample.devices:
            by_gpu[device.index].append(device)

    findings: list[str] = []
    summaries: list[GPUTimeSeriesSummary] = []

    mean_utils: dict[int, float] = {}
    for index, devices in sorted(by_gpu.items()):
        util = [
            float(device.gpu_util_pct)
            for device in devices
            if device.gpu_util_pct is not None
        ]
        memory = [
            float(device.memory_pressure_pct)
            for device in devices
            if device.memory_pressure_pct is not None
        ]
        power = [
            float(device.power_draw_w)
            for device in devices
            if device.power_draw_w is not None
        ]
        temperature = [
            float(device.temperature_c)
            for device in devices
            if device.temperature_c is not None
        ]
        util_mean = _avg(util)
        if util_mean is not None:
            mean_utils[index] = util_mean

        summaries.append(
            GPUTimeSeriesSummary(
                gpu_index=index,
                sample_count=len(devices),
                gpu_util_mean_pct=util_mean,
                gpu_util_min_pct=min(util) if util else None,
                gpu_util_max_pct=max(util) if util else None,
                memory_pressure_mean_pct=_avg(memory),
                memory_pressure_max_pct=_max(memory),
                power_draw_mean_w=_avg(power),
                temperature_max_c=_max(temperature),
            )
        )

    stragglers: list[int] = []
    reference: float | None = None
    if len(mean_utils) >= 2:
        values = list(mean_utils.values())
        reference = median(values) if len(values) >= 3 else max(values)

        updated: list[GPUTimeSeriesSummary] = []
        for summary in summaries:
            gap = None
            if summary.gpu_util_mean_pct is not None:
                gap = max(0.0, reference - summary.gpu_util_mean_pct)
                if gap >= straggler_gap_pct_points:
                    stragglers.append(summary.gpu_index)

            updated.append(
                GPUTimeSeriesSummary(
                    gpu_index=summary.gpu_index,
                    sample_count=summary.sample_count,
                    gpu_util_mean_pct=summary.gpu_util_mean_pct,
                    gpu_util_min_pct=summary.gpu_util_min_pct,
                    gpu_util_max_pct=summary.gpu_util_max_pct,
                    memory_pressure_mean_pct=summary.memory_pressure_mean_pct,
                    memory_pressure_max_pct=summary.memory_pressure_max_pct,
                    power_draw_mean_w=summary.power_draw_mean_w,
                    temperature_max_c=summary.temperature_max_c,
                    straggler_gap_pct_points=gap,
                )
            )
        summaries = updated

    if not by_gpu:
        findings.append(
            "Profile contains aggregate-only telemetry; per-GPU time-series analysis "
            "requires a profile collected with v0.8 or later."
        )
    elif stragglers:
        rendered = ", ".join(str(index) for index in stragglers)
        findings.append(
            f"Persistent utilization straggler signal detected on GPU(s) {rendered} "
            f"using a {straggler_gap_pct_points:.1f}-point mean-utilization gap."
        )

    high_memory = [
        summary.gpu_index
        for summary in summaries
        if summary.memory_pressure_max_pct is not None
        and summary.memory_pressure_max_pct >= 95.0
    ]
    if high_memory:
        findings.append(
            "Peak memory pressure reached at least 95% on GPU(s) "
            + ", ".join(str(index) for index in high_memory)
            + "."
        )

    dcgm_summary = summarize_dcgm(dcgm_frames or [])
    if dcgm_summary:
        dcgm_util = {
            index: metrics["DCGM_FI_DEV_GPU_UTIL"]
            for index, metrics in dcgm_summary.items()
            if "DCGM_FI_DEV_GPU_UTIL" in metrics
        }
        if len(dcgm_util) >= 2:
            values = list(dcgm_util.values())
            dcgm_reference = median(values) if len(values) >= 3 else max(values)
            slow = [
                index
                for index, value in dcgm_util.items()
                if dcgm_reference - value >= straggler_gap_pct_points
            ]
            if slow:
                findings.append(
                    "DCGM GPU-utilization imbalance independently flags GPU(s) "
                    + ", ".join(str(index) for index in slow)
                    + "."
                )

    phase_summaries = [
        _phase_summary(profile, phase)
        for phase in (phases or [])
    ]

    return TimeSeriesReport(
        profile_id=profile.id,
        gpu_summaries=summaries,
        straggler_gpu_indices=stragglers,
        phases=phase_summaries,
        dcgm_summary=dcgm_summary,
        findings=findings,
    )
