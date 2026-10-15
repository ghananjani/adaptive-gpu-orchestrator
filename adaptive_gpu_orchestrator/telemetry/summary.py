from __future__ import annotations

from statistics import mean, pstdev

from adaptive_gpu_orchestrator.models import GPUDevice


def _values(gpus: list[GPUDevice], attr: str) -> list[float]:
    values: list[float] = []
    for gpu in gpus:
        value = getattr(gpu, attr)
        if value is not None:
            values.append(float(value))
    return values


def summarize_utilization(gpus: list[GPUDevice]) -> dict[str, float | None]:
    util = _values(gpus, "utilization_gpu_pct")
    mem = [x for x in (gpu.memory_pressure_pct for gpu in gpus) if x is not None]
    temp = _values(gpus, "temperature_c")
    power_ratio = [
        100.0 * gpu.power_draw_w / gpu.power_limit_w
        for gpu in gpus
        if gpu.power_draw_w is not None and gpu.power_limit_w not in (None, 0)
    ]
    return {
        "gpu_util_mean_pct": mean(util) if util else None,
        "gpu_util_std_pct": pstdev(util) if len(util) > 1 else 0.0 if util else None,
        "memory_pressure_mean_pct": mean(mem) if mem else None,
        "temperature_max_c": max(temp) if temp else None,
        "power_ratio_max_pct": max(power_ratio) if power_ratio else None,
    }
