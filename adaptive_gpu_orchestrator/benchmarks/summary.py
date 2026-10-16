from __future__ import annotations

from statistics import mean

from adaptive_gpu_orchestrator.models import NCCLResult


def summarize_nccl(
    results: list[NCCLResult] | tuple[NCCLResult, ...],
) -> dict[str, float | int | None]:
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
