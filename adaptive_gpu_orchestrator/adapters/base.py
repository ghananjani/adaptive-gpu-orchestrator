from __future__ import annotations

from typing import Any, Protocol

from adaptive_gpu_orchestrator.optimizer.trials import TrialMetrics


class MetricsAdapter(Protocol):
    name: str

    def parse(self, payload: dict[str, Any]) -> TrialMetrics:
        ...


def numeric_extras(
    payload: dict[str, Any],
    *,
    prefixes: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
) -> dict[str, float]:
    extras: dict[str, float] = {}

    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            extras[key] = float(value)

    for prefix in prefixes:
        nested = payload.get(prefix)
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            if isinstance(value, (int, float)):
                extras[f"{prefix}.{key}"] = float(value)

    return extras
