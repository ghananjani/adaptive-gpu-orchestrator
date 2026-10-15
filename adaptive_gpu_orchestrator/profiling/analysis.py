from __future__ import annotations

from adaptive_gpu_orchestrator.models import (
    Bottleneck,
    DiagnosisEvidence,
    DiagnosisReport,
    Recommendation,
)
from adaptive_gpu_orchestrator.profiling.models import WorkloadProfile


def diagnose_profile(profile: WorkloadProfile) -> DiagnosisReport:
    summary = profile.summary()
    util = summary["gpu_util_mean_pct"]
    imbalance = summary["gpu_util_std_pct"]
    memory = summary["memory_pressure_mean_pct"]
    temperature = summary["temperature_max_c"]

    if not profile.samples or summary["gpu_count_max"] == 0:
        return DiagnosisReport(
            bottleneck=Bottleneck.NO_GPU,
            severity="info",
            summary="The workload profile contains no GPU telemetry.",
        )

    if temperature is not None and temperature >= 85:
        return DiagnosisReport(
            bottleneck=Bottleneck.THERMAL_OR_POWER,
            severity="high",
            summary="The workload reached a high GPU temperature during profiling.",
            evidence=[
                DiagnosisEvidence(
                    "temperature_max_c",
                    f"{temperature:.1f}",
                    "Sustained high temperature can trigger clock throttling.",
                )
            ],
            recommendations=[
                Recommendation(
                    "high",
                    "Inspect clocks, cooling and power limits",
                    "Thermal throttling can mask the workload's true scaling behavior.",
                )
            ],
        )

    if imbalance is not None and imbalance >= 20:
        return DiagnosisReport(
            bottleneck=Bottleneck.LOAD_IMBALANCE,
            severity="high",
            summary="The workload shows sustained utilization imbalance across GPUs.",
            evidence=[
                DiagnosisEvidence(
                    "gpu_util_std_pct",
                    f"{imbalance:.1f}%",
                    "High cross-GPU variance suggests rank imbalance or stragglers.",
                )
            ],
            recommendations=[
                Recommendation(
                    "high",
                    "Inspect per-rank work and placement",
                    "Compare rank step times and map communication-heavy ranks to topology.",
                )
            ],
        )

    if memory is not None and memory >= 90 and (util is None or util < 90):
        return DiagnosisReport(
            bottleneck=Bottleneck.MEMORY_PRESSURE,
            severity="high",
            summary="The workload is persistently memory-heavy without saturating compute.",
            evidence=[
                DiagnosisEvidence(
                    "memory_pressure_mean_pct",
                    f"{memory:.1f}%",
                    "High VRAM occupancy can constrain batching, KV cache, and parallelism.",
                )
            ],
            recommendations=[
                Recommendation(
                    "high",
                    "Profile allocation and cache behavior",
                    "Evaluate KV-cache policy, sharding, checkpointing, and batch shape.",
                )
            ],
        )

    if util is not None and util < 50 and (memory is None or memory < 80):
        return DiagnosisReport(
            bottleneck=Bottleneck.UNDERUTILIZED,
            severity="medium",
            summary="The profiled workload leaves substantial GPU compute capacity idle.",
            evidence=[
                DiagnosisEvidence(
                    "gpu_util_mean_pct",
                    f"{util:.1f}%",
                    "Low average utilization persisted across the profiling window.",
                )
            ],
            recommendations=[
                Recommendation(
                    "high",
                    "Measure CPU, I/O and collective wait time",
                    "Low utilization commonly reflects input stalls, synchronization, "
                    "communication, or insufficient batching.",
                )
            ],
        )

    return DiagnosisReport(
        bottleneck=Bottleneck.HEALTHY,
        severity="info",
        summary="No high-confidence workload-level bottleneck was detected.",
        evidence=[
            DiagnosisEvidence(
                "gpu_util_mean_pct",
                f"{util:.1f}%" if util is not None else "unknown",
                "Average utilization over the complete workload profile.",
            )
        ],
    )
