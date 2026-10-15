from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from adaptive_gpu_orchestrator.profiling.models import WorkloadProfile
from adaptive_gpu_orchestrator.telemetry.providers import NvidiaSMIProvider, TelemetryProvider
from adaptive_gpu_orchestrator.telemetry.sampler import TelemetrySampler


@dataclass(slots=True)
class WorkloadProfiler:
    provider: TelemetryProvider = field(default_factory=NvidiaSMIProvider)
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep

    def run(
        self,
        *,
        name: str,
        command: list[str] | None = None,
        duration_s: float = 10.0,
        interval_s: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> WorkloadProfile:
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")

        profile = WorkloadProfile(name=name, command=command or [], tags=tags or {})
        sampler = TelemetrySampler(
            self.provider,
            clock=self.clock,
            sleeper=self.sleeper,
        )
        started = self.clock()
        process = subprocess.Popen(command) if command else None

        try:
            while True:
                elapsed = self.clock() - started
                profile.samples.append(sampler.sample_once(elapsed))

                if process is not None and process.poll() is not None:
                    profile.exit_code = process.returncode
                    break
                if process is None and elapsed >= duration_s:
                    break

                if process is not None and duration_s > 0 and elapsed >= duration_s:
                    process.terminate()
                    try:
                        profile.exit_code = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        profile.exit_code = process.wait()
                    break

                remaining = (
                    interval_s
                    if process is not None or duration_s <= 0
                    else min(interval_s, max(0.0, duration_s - elapsed))
                )
                self.sleeper(remaining)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        profile.duration_s = max(0.0, self.clock() - started)
        profile.finished_at = datetime.now(timezone.utc).isoformat()
        return profile
