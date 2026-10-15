from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

_PROMETHEUS_METRIC = re.compile(
    r"^(?P<name>DCGM_[A-Z0-9_]+)\{(?P<labels>[^}]*)\}\s+"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)$"
)
_LABEL = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:\\.|[^"])*)"')

KNOWN_EXPORTER_FIELDS = {
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_MEM_COPY_UTIL",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_FB_FREE",
    "DCGM_FI_DEV_FB_TOTAL",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_MEMORY_TEMP",
    "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL",
}


@dataclass(frozen=True, slots=True)
class DCGMDeviceSample:
    offset_s: float
    gpu_index: int
    uuid: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_s": self.offset_s,
            "gpu_index": self.gpu_index,
            "uuid": self.uuid,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class DCGMFrame:
    offset_s: float
    devices: tuple[DCGMDeviceSample, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_s": self.offset_s,
            "devices": [device.to_dict() for device in self.devices],
        }


def _labels(text: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value").replace(r'\"', '"')
        for match in _LABEL.finditer(text)
    }


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _identity(labels: dict[str, str]) -> tuple[int, str]:
    raw_index = labels.get("gpu") or labels.get("GPU") or labels.get("device") or "0"
    try:
        gpu_index = int(raw_index)
    except ValueError:
        match = re.search(r"\d+", raw_index)
        gpu_index = int(match.group()) if match else 0
    uuid = labels.get("UUID") or labels.get("uuid") or ""
    return gpu_index, uuid


def parse_dcgm_exporter(
    text: str,
    *,
    interval_s: float = 1.0,
) -> list[DCGMFrame]:
    """Parse one or more concatenated DCGM Exporter Prometheus scrapes.

    A new frame is inferred when the same GPU/metric pair appears again.
    Unknown numeric DCGM metrics are preserved alongside known fields.
    """

    if interval_s <= 0:
        raise ValueError("interval_s must be positive")

    frames: list[DCGMFrame] = []
    current: dict[tuple[int, str], dict[str, float]] = {}
    seen: set[tuple[int, str, str]] = set()
    frame_index = 0

    def flush() -> None:
        nonlocal current, seen, frame_index
        if not current:
            return
        offset = frame_index * interval_s
        devices = tuple(
            DCGMDeviceSample(
                offset_s=offset,
                gpu_index=index,
                uuid=uuid,
                metrics=dict(metrics),
            )
            for (index, uuid), metrics in sorted(current.items())
        )
        frames.append(DCGMFrame(offset_s=offset, devices=devices))
        current = {}
        seen = set()
        frame_index += 1

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_METRIC.match(line)
        if not match:
            continue

        value = _number(match.group("value"))
        if value is None:
            continue

        labels = _labels(match.group("labels"))
        gpu_index, uuid = _identity(labels)
        metric = match.group("name")
        occurrence = (gpu_index, uuid, metric)

        if occurrence in seen:
            flush()

        seen.add(occurrence)
        current.setdefault((gpu_index, uuid), {})[metric] = value

    flush()
    return frames


def parse_dcgm_dmon(
    text: str,
    *,
    interval_s: float = 1.0,
) -> list[DCGMFrame]:
    """Parse dcgmi dmon tabular output into per-GPU frames."""

    if interval_s <= 0:
        raise ValueError("interval_s must be positive")

    headers: list[str] = []
    frames: list[DCGMFrame] = []
    current: dict[int, dict[str, float]] = {}
    frame_index = 0

    def flush() -> None:
        nonlocal current, frame_index
        if not current:
            return
        offset = frame_index * interval_s
        frames.append(
            DCGMFrame(
                offset_s=offset,
                devices=tuple(
                    DCGMDeviceSample(
                        offset_s=offset,
                        gpu_index=index,
                        metrics=dict(metrics),
                    )
                    for index, metrics in sorted(current.items())
                ),
            )
        )
        current = {}
        frame_index += 1

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#") and "Entity" in line:
            parts = line.lstrip("#").split()
            try:
                entity_index = parts.index("Entity")
            except ValueError:
                continue
            headers = parts[entity_index + 1 :]
            continue

        parts = line.split()
        if len(parts) < 3 or parts[0].upper() != "GPU":
            continue
        if not headers:
            continue

        try:
            gpu_index = int(parts[1])
        except ValueError:
            continue

        if gpu_index in current:
            flush()

        values = parts[2:]
        metrics: dict[str, float] = {}
        for header, raw_value in zip(headers, values, strict=False):
            value = _number(raw_value)
            if value is not None:
                metrics[header] = value
        current[gpu_index] = metrics

    flush()
    return frames


def summarize_dcgm(
    frames: list[DCGMFrame],
) -> dict[int, dict[str, float]]:
    by_gpu: dict[int, dict[str, list[float]]] = {}
    for frame in frames:
        for device in frame.devices:
            metrics = by_gpu.setdefault(device.gpu_index, {})
            for name, value in device.metrics.items():
                metrics.setdefault(name, []).append(float(value))

    return {
        gpu_index: {
            name: mean(values)
            for name, values in sorted(metrics.items())
            if values
        }
        for gpu_index, metrics in sorted(by_gpu.items())
    }
