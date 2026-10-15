from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from adaptive_gpu_orchestrator.models import NCCLResult

_DATA_ROW = re.compile(
    r"^\s*(?P<size>\d+)\s+\d+\s+\w+\s+\w+\s+-?\d+\s+"
    r"(?P<time>[\d.]+)\s+(?P<algbw>[\d.]+)\s+(?P<busbw>[\d.]+)"
)


def parse_nccl_output(text: str, operation: str = "all_reduce") -> list[NCCLResult]:
    results: list[NCCLResult] = []
    for line in text.splitlines():
        match = _DATA_ROW.match(line)
        if not match:
            continue
        results.append(
            NCCLResult(
                operation=operation,
                size_bytes=int(match.group("size")),
                time_us=float(match.group("time")),
                algbw_gbps=float(match.group("algbw")),
                busbw_gbps=float(match.group("busbw")),
            )
        )
    return results


@dataclass(slots=True)
class NCCLBenchmark:
    binary: str = "all_reduce_perf"

    def run(self, begin: str = "8M", end: str = "1G", factor: int = 2, gpus: int = 8):
        path = Path(self.binary)
        executable = str(path) if path.exists() else self.binary
        command = [executable, "-b", begin, "-e", end, "-f", str(factor), "-g", str(gpus)]
        proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=180)
        if proc.returncode != 0:
            rendered = " ".join(shlex.quote(part) for part in command)
            raise RuntimeError(proc.stderr.strip() or f"NCCL benchmark failed: {rendered}")
        return parse_nccl_output(proc.stdout)
