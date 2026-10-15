from __future__ import annotations

import csv
import io
import re
import shutil
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from adaptive_gpu_orchestrator.models import ClusterSnapshot, GPUDevice

from .topology import parse_topology_matrix


class NvidiaSMIError(RuntimeError):
    pass


Runner = Callable[[list[str]], str]


def _default_runner(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NvidiaSMIError(str(exc)) from exc
    if result.returncode != 0:
        raise NvidiaSMIError(result.stderr.strip() or f"command failed: {' '.join(args)}")
    return result.stdout


def _number(value: str) -> float | None:
    value = value.strip()
    if value in {"", "N/A", "[Not Supported]", "Not Supported"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


@dataclass(slots=True)
class NvidiaSMI:
    runner: Runner = _default_runner

    @staticmethod
    def available() -> bool:
        return shutil.which("nvidia-smi") is not None

    def _run(self, *args: str) -> str:
        return self.runner(["nvidia-smi", *args])

    def query_gpus(self) -> list[GPUDevice]:
        fields = [
            "index", "name", "uuid", "memory.total", "memory.used",
            "utilization.gpu", "utilization.memory", "temperature.gpu",
            "power.draw", "power.limit", "pci.bus_id", "compute_cap",
        ]
        output = self._run(
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
        rows = csv.reader(io.StringIO(output), skipinitialspace=True)
        gpus: list[GPUDevice] = []
        for row in rows:
            if len(row) < len(fields):
                continue
            gpus.append(
                GPUDevice(
                    index=int(row[0].strip()),
                    name=row[1].strip(),
                    uuid=row[2].strip(),
                    memory_total_mib=int(_number(row[3]) or 0),
                    memory_used_mib=int(_number(row[4]) or 0),
                    utilization_gpu_pct=_number(row[5]),
                    utilization_memory_pct=_number(row[6]),
                    temperature_c=_number(row[7]),
                    power_draw_w=_number(row[8]),
                    power_limit_w=_number(row[9]),
                    pci_bus_id=row[10].strip(),
                    compute_capability=row[11].strip(),
                )
            )
        return gpus

    def topology(self):
        return parse_topology_matrix(self._run("topo", "-m"))

    def driver_version(self) -> str | None:
        output = self._run("--query-gpu=driver_version", "--format=csv,noheader")
        line = output.strip().splitlines()
        return line[0].strip() if line else None

    def cuda_version(self) -> str | None:
        output = self._run()
        match = re.search(r"CUDA Version:\s*([\d.]+)", output)
        return match.group(1) if match else None

    def snapshot(self) -> ClusterSnapshot:
        warnings: list[str] = []
        if not self.available() and self.runner is _default_runner:
            return ClusterSnapshot(
                hostname=socket.gethostname(),
                warnings=["nvidia-smi not found; running in no-GPU mode"],
            )

        try:
            gpus = self.query_gpus()
        except NvidiaSMIError as exc:
            return ClusterSnapshot(hostname=socket.gethostname(), warnings=[str(exc)])

        try:
            topology = self.topology()
        except NvidiaSMIError as exc:
            topology = []
            warnings.append(f"topology unavailable: {exc}")

        try:
            driver = self.driver_version()
        except NvidiaSMIError:
            driver = None
        try:
            cuda = self.cuda_version()
        except NvidiaSMIError:
            cuda = None

        return ClusterSnapshot(
            hostname=socket.gethostname(),
            gpus=gpus,
            topology=topology,
            driver_version=driver,
            cuda_version=cuda,
            warnings=warnings,
        )
