from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol

from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig
from adaptive_gpu_orchestrator.optimizer.trials import TrialMetrics


class CanaryRunner(Protocol):
    def run(self, config: ParallelismConfig) -> TrialMetrics:
        ...


@dataclass(slots=True)
class CommandCanaryRunner:
    """Runs a benchmark command and parses its final JSON metrics object.

    The child process receives configuration through environment variables:
    AGO_TP, AGO_PP, AGO_DP, AGO_BATCH_SIZE, AGO_CONCURRENCY.

    The benchmark should emit a JSON object containing any supported metrics,
    for example:
    {"throughput_per_s": 1200, "p99_latency_ms": 85, "energy_joules": 4200}
    """

    command: list[str]
    timeout_s: float = 120.0
    extra_env: dict[str, str] | None = None

    def _environment(self, config: ParallelismConfig) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.extra_env or {})
        env.update(
            {
                "AGO_TP": str(config.tensor_parallel),
                "AGO_PP": str(config.pipeline_parallel),
                "AGO_DP": str(config.data_parallel),
                "AGO_BATCH_SIZE": str(config.batch_size),
                "AGO_CONCURRENCY": str(config.concurrency),
            }
        )
        return env

    @staticmethod
    def _parse_metrics(stdout: str) -> dict:
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("benchmark did not emit a JSON metrics object")

    def run(self, config: ParallelismConfig) -> TrialMetrics:
        if not self.command:
            raise ValueError("benchmark command cannot be empty")

        started = time.monotonic()
        try:
            completed = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                env=self._environment(config),
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - started
            return TrialMetrics(duration_s=duration, exit_code=124)

        duration = time.monotonic() - started
        if completed.returncode != 0:
            return TrialMetrics(duration_s=duration, exit_code=completed.returncode)

        payload = self._parse_metrics(completed.stdout)
        payload.setdefault("duration_s", duration)
        payload.setdefault("exit_code", completed.returncode)
        return TrialMetrics.from_dict(payload)
