from __future__ import annotations

from dataclasses import dataclass

from adaptive_gpu_orchestrator.models import GPUDevice


class NVMLUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class NVMLProvider:
    """Optional low-overhead telemetry provider backed by nvidia-ml-py."""

    initialized: bool = False

    def _module(self):
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise NVMLUnavailable(
                "NVML telemetry requires the optional 'nvml' extra: "
                "pip install 'adaptive-gpu-orchestrator[nvml]'"
            ) from exc
        return pynvml

    def _ensure_initialized(self):
        pynvml = self._module()
        if not self.initialized:
            pynvml.nvmlInit()
            self.initialized = True
        return pynvml

    @staticmethod
    def _decode(value) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    def sample(self) -> list[GPUDevice]:
        pynvml = self._ensure_initialized()
        devices: list[GPUDevice] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)

            def optional(call, divisor: float = 1.0):
                try:
                    return float(call()) / divisor
                except pynvml.NVMLError:
                    return None

            power_draw = optional(lambda: pynvml.nvmlDeviceGetPowerUsage(handle), 1000.0)
            power_limit = optional(
                lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle),
                1000.0,
            )
            temperature = optional(
                lambda: pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            )

            devices.append(
                GPUDevice(
                    index=index,
                    name=self._decode(pynvml.nvmlDeviceGetName(handle)),
                    uuid=self._decode(pynvml.nvmlDeviceGetUUID(handle)),
                    memory_total_mib=int(memory.total / (1024 * 1024)),
                    memory_used_mib=int(memory.used / (1024 * 1024)),
                    utilization_gpu_pct=float(util.gpu),
                    utilization_memory_pct=float(util.memory),
                    temperature_c=temperature,
                    power_draw_w=power_draw,
                    power_limit_w=power_limit,
                    pci_bus_id=self._decode(pci.busId),
                )
            )
        return devices

    def close(self) -> None:
        if not self.initialized:
            return
        pynvml = self._module()
        pynvml.nvmlShutdown()
        self.initialized = False
