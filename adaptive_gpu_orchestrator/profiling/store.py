from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adaptive_gpu_orchestrator.profiling.models import WorkloadProfile


@dataclass(slots=True)
class ExperimentStore:
    root: Path = Path(".ago/experiments")

    def save(self, profile: WorkloadProfile) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{profile.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(profile.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, identifier: str | Path) -> WorkloadProfile:
        candidate = Path(identifier)
        path = candidate if candidate.exists() else self.root / f"{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(f"experiment profile not found: {identifier}")
        return WorkloadProfile.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def list(self) -> list[WorkloadProfile]:
        if not self.root.exists():
            return []
        profiles: list[WorkloadProfile] = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                profiles.append(self.load(path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return profiles
