from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, replace

from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig
from adaptive_gpu_orchestrator.optimizer.trials import TrialMetrics

from .base import MetricsAdapter


@dataclass(slots=True)
class AdaptedCommandCanaryRunner:
    adapter: MetricsAdapter
    command: list[str]
    timeout_s: float = 120.0
    extra_env: dict[str, str] | None = None

    def _environment(self, config: ParallelismConfig) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.extra_env or {})
        env.update(
            {
                "AGO_TP": str(config.tensor_parallel),
                "AGO_TENSOR_PARALLEL": str(config.tensor_parallel),
                "AGO_PP": str(config.pipeline_parallel),
                "AGO_PIPELINE_PARALLEL": str(config.pipeline_parallel),
                "AGO_DP": str(config.data_parallel),
                "AGO_DATA_PARALLEL": str(config.data_parallel),
                "AGO_BATCH_SIZE": str(config.batch_size),
                "AGO_CONCURRENCY": str(config.concurrency),
                "AGO_WORLD_SIZE": str(config.world_size),
                "AGO_CONFIG_LABEL": config.label(),
                "AGO_METRICS_ADAPTER": self.adapter.name,
            }
        )
        return env

    @staticmethod
    def _parse_payload(stdout: str) -> dict:
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
            return TrialMetrics(
                duration_s=time.monotonic() - started,
                exit_code=124,
            )

        duration = time.monotonic() - started
        if completed.returncode != 0:
            return TrialMetrics(
                duration_s=duration,
                exit_code=completed.returncode,
            )

        metrics = self.adapter.parse(self._parse_payload(completed.stdout))
        return replace(
            metrics,
            duration_s=metrics.duration_s if metrics.duration_s is not None else duration,
            exit_code=completed.returncode,
        )
