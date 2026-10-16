from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import PerformanceModelReport


@dataclass(slots=True)
class PerformanceModelStore:
    root: Path = Path(".ago/models")

    def save(self, report: PerformanceModelReport) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{report.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, identifier: str | Path) -> PerformanceModelReport:
        candidate = Path(identifier)
        path = candidate if candidate.exists() else self.root / f"{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(f"performance model not found: {identifier}")
        return PerformanceModelReport.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def list(self) -> list[PerformanceModelReport]:
        if not self.root.exists():
            return []

        reports = []
        paths = sorted(
            self.root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                reports.append(self.load(path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return reports
