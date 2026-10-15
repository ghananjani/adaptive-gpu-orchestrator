from __future__ import annotations

from adaptive_gpu_orchestrator.models import (
    Bottleneck,
    ClusterSnapshot,
    DiagnosisEvidence,
    DiagnosisReport,
    NCCLResult,
    Recommendation,
)
from adaptive_gpu_orchestrator.telemetry import summarize_utilization


def diagnose(snapshot: ClusterSnapshot, nccl: list[NCCLResult] | None = None) -> DiagnosisReport:
    if not snapshot.gpus:
        return DiagnosisReport(
            bottleneck=Bottleneck.NO_GPU,
            severity="info",
            summary="No NVIDIA GPUs were discovered on this host.",
            recommendations=[
                Recommendation("low", "Install/verify NVIDIA drivers", "agoctl uses nvidia-smi for v0.1 discovery."),
            ],
        )

    metrics = summarize_utilization(snapshot.gpus)
    util = metrics["gpu_util_mean_pct"]
    util_std = metrics["gpu_util_std_pct"]
    mem = metrics["memory_pressure_mean_pct"]
    temp = metrics["temperature_max_c"]
    power = metrics["power_ratio_max_pct"]

    if (temp is not None and temp >= 85) or (power is not None and power >= 98):
        return DiagnosisReport(
            bottleneck=Bottleneck.THERMAL_OR_POWER,
            severity="high",
            summary="At least one GPU is operating close to a thermal or power ceiling.",
            evidence=[
                DiagnosisEvidence("temperature_max_c", f"{temp}", "High temperature can trigger clock throttling."),
                DiagnosisEvidence("power_ratio_max_pct", f"{power}", "Operating at the power limit can cap performance."),
            ],
            recommendations=[
                Recommendation("high", "Check throttling reasons and cooling", "Validate nvidia-smi clocks, thermals, and power caps."),
            ],
        )

    if util_std is not None and util_std >= 20:
        return DiagnosisReport(
            bottleneck=Bottleneck.LOAD_IMBALANCE,
            severity="high",
            summary="GPU utilization is imbalanced across devices.",
            evidence=[DiagnosisEvidence("gpu_util_std_pct", f"{util_std:.1f}%", "Large per-GPU variance suggests stragglers or uneven placement.")],
            recommendations=[
                Recommendation("high", "Inspect rank placement and per-rank step time", "Map hot ranks to topology and verify equal work partitioning."),
                Recommendation("medium", "Profile dataloading and collectives", "Stragglers can come from CPU, I/O, or communication imbalance."),
            ],
        )

    if mem is not None and mem >= 90 and (util is None or util < 90):
        return DiagnosisReport(
            bottleneck=Bottleneck.MEMORY_PRESSURE,
            severity="high",
            summary="GPU memory pressure is high relative to compute utilization.",
            evidence=[
                DiagnosisEvidence("memory_pressure_mean_pct", f"{mem:.1f}%", "High VRAM occupancy leaves little room for batching/KV growth."),
                DiagnosisEvidence("gpu_util_mean_pct", f"{util}", "Compute is not fully saturated."),
            ],
            recommendations=[
                Recommendation("high", "Reduce memory pressure", "Evaluate KV cache policy, quantization, checkpointing, sharding, or batch shape."),
            ],
        )

    if nccl:
        peak_bus = max(result.busbw_gbps for result in nccl)
        if util is not None and util < 70 and peak_bus < 100:
            return DiagnosisReport(
                bottleneck=Bottleneck.COMMUNICATION_BOUND,
                severity="medium",
                summary="Low GPU utilization coincides with weak collective bandwidth.",
                evidence=[
                    DiagnosisEvidence("gpu_util_mean_pct", f"{util:.1f}%", "Compute resources are under-filled."),
                    DiagnosisEvidence("nccl_peak_busbw_gbps", f"{peak_bus:.1f}", "Collective bandwidth is below the v0.1 warning threshold."),
                ],
                recommendations=[
                    Recommendation("high", "Validate NCCL topology and rank placement", "Keep communication-heavy ranks within fast NVLink/NVSwitch domains where possible."),
                    Recommendation("medium", "Profile collective time", "Measure all-reduce/all-gather contribution to iteration latency."),
                ],
            )

    if util is not None and util < 50 and (mem is None or mem < 80):
        return DiagnosisReport(
            bottleneck=Bottleneck.UNDERUTILIZED,
            severity="medium",
            summary="GPUs are substantially underutilized without obvious VRAM pressure.",
            evidence=[DiagnosisEvidence("gpu_util_mean_pct", f"{util:.1f}%", "Low sustained compute utilization.")],
            recommendations=[
                Recommendation("high", "Profile CPU/I/O/communication stalls", "Check dataloading, synchronization, kernel launch gaps, and network collectives."),
                Recommendation("medium", "Increase useful work per launch", "Test larger batches, CUDA graphs, or better request batching when workload semantics permit."),
            ],
        )

    return DiagnosisReport(
        bottleneck=Bottleneck.HEALTHY,
        severity="info",
        summary="No high-confidence v0.1 bottleneck was detected from the available signals.",
        evidence=[DiagnosisEvidence("gpu_util_mean_pct", f"{util}", "Current mean compute utilization.")],
        recommendations=[Recommendation("low", "Capture a workload profile", "Use PyTorch/vLLM profiling in a future release for phase-aware diagnosis.")],
    )
