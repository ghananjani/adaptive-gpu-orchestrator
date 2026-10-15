from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from adaptive_gpu_orchestrator.models import GPUDevice
from adaptive_gpu_orchestrator.telemetry.providers import TelemetryProvider
from adaptive_gpu_orchestrator.telemetry.summary import summarize_utilization


@dataclass(frozen=True, slots=True)
class DeviceTelemetry:
    index: int
    uuid: str = ""
    gpu_util_pct: float | None = None
    memory_copy_util_pct: float | None = None
    memory_pressure_pct: float | None = None
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None

    @classmethod
    def from_gpu(cls, gpu: GPUDevice) -> DeviceTelemetry:
        return cls(
            index=gpu.index,
            uuid=gpu.uuid,
            gpu_util_pct=gpu.utilization_gpu_pct,
            memory_copy_util_pct=gpu.utilization_memory_pct,
            memory_pressure_pct=gpu.memory_pressure_pct,
            memory_used_mib=gpu.memory_used_mib,
            memory_total_mib=gpu.memory_total_mib,
            temperature_c=gpu.temperature_c,
            power_draw_w=gpu.power_draw_w,
            power_limit_w=gpu.power_limit_w,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "gpu_util_pct": self.gpu_util_pct,
            "memory_copy_util_pct": self.memory_copy_util_pct,
            "memory_pressure_pct": self.memory_pressure_pct,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
            "power_limit_w": self.power_limit_w,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceTelemetry:
        return cls(
            index=int(data["index"]),
            uuid=str(data.get("uuid", "")),
            gpu_util_pct=data.get("gpu_util_pct"),
            memory_copy_util_pct=data.get("memory_copy_util_pct"),
            memory_pressure_pct=data.get("memory_pressure_pct"),
            memory_used_mib=data.get("memory_used_mib"),
            memory_total_mib=data.get("memory_total_mib"),
            temperature_c=data.get("temperature_c"),
            power_draw_w=data.get("power_draw_w"),
            power_limit_w=data.get("power_limit_w"),
        )


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    offset_s: float
    gpu_count: int
    gpu_util_mean_pct: float | None
    gpu_util_std_pct: float | None
    memory_pressure_mean_pct: float | None
    temperature_max_c: float | None
    power_ratio_max_pct: float | None
    power_draw_total_w: float | None
    devices: tuple[DeviceTelemetry, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_s": self.offset_s,
            "gpu_count": self.gpu_count,
            "gpu_util_mean_pct": self.gpu_util_mean_pct,
            "gpu_util_std_pct": self.gpu_util_std_pct,
            "memory_pressure_mean_pct": self.memory_pressure_mean_pct,
            "temperature_max_c": self.temperature_max_c,
            "power_ratio_max_pct": self.power_ratio_max_pct,
            "power_draw_total_w": self.power_draw_total_w,
            "devices": [device.to_dict() for device in self.devices],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelemetrySample:
        return cls(
            offset_s=float(data["offset_s"]),
            gpu_count=int(data["gpu_count"]),
            gpu_util_mean_pct=data.get("gpu_util_mean_pct"),
            gpu_util_std_pct=data.get("gpu_util_std_pct"),
            memory_pressure_mean_pct=data.get("memory_pressure_mean_pct"),
            temperature_max_c=data.get("temperature_max_c"),
            power_ratio_max_pct=data.get("power_ratio_max_pct"),
            power_draw_total_w=data.get("power_draw_total_w"),
            devices=tuple(
                DeviceTelemetry.from_dict(device)
                for device in data.get("devices", [])
            ),
        )


class TelemetrySampler:
    def __init__(
        self,
        provider: TelemetryProvider,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.clock = clock
        self.sleeper = sleeper

    def sample_once(self, offset_s: float = 0.0) -> TelemetrySample:
        gpus = self.provider.sample()
        summary = summarize_utilization(gpus)
        power_values = [gpu.power_draw_w for gpu in gpus if gpu.power_draw_w is not None]
        return TelemetrySample(
            offset_s=offset_s,
            gpu_count=len(gpus),
            gpu_util_mean_pct=summary["gpu_util_mean_pct"],
            gpu_util_std_pct=summary["gpu_util_std_pct"],
            memory_pressure_mean_pct=summary["memory_pressure_mean_pct"],
            temperature_max_c=summary["temperature_max_c"],
            power_ratio_max_pct=summary["power_ratio_max_pct"],
            power_draw_total_w=sum(power_values) if power_values else None,
            devices=tuple(DeviceTelemetry.from_gpu(gpu) for gpu in gpus),
        )

    def collect(self, duration_s: float, interval_s: float = 1.0) -> list[TelemetrySample]:
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")

        start = self.clock()
        samples: list[TelemetrySample] = []
        while True:
            elapsed = self.clock() - start
            samples.append(self.sample_once(elapsed))
            if elapsed >= duration_s:
                break
            self.sleeper(min(interval_s, max(0.0, duration_s - elapsed)))
        return samples


def average_sample(samples: list[TelemetrySample]) -> dict[str, float | None]:
    def avg(attr: str) -> float | None:
        values = [getattr(sample, attr) for sample in samples]
        numeric = [float(value) for value in values if value is not None]
        return mean(numeric) if numeric else None

    return {
        "gpu_util_mean_pct": avg("gpu_util_mean_pct"),
        "gpu_util_std_pct": avg("gpu_util_std_pct"),
        "memory_pressure_mean_pct": avg("memory_pressure_mean_pct"),
        "temperature_max_c": max(
            (
                sample.temperature_max_c
                for sample in samples
                if sample.temperature_max_c is not None
            ),
            default=None,
        ),
        "power_draw_mean_w": avg("power_draw_total_w"),
    }
