from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import CorrelationReport


@dataclass(slots=True)
class CorrelationStore:
    root: Path = Path(".ago/correlations")

    def save(self, report: CorrelationReport) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{report.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, identifier: str | Path) -> CorrelationReport:
        candidate = Path(identifier)
        path = candidate if candidate.exists() else self.root / f"{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(f"correlation report not found: {identifier}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return CorrelationReport.from_dict(data)

    def list(self) -> list[CorrelationReport]:
        if not self.root.exists():
            return []
        paths = sorted(
            self.root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        reports: list[CorrelationReport] = []
        for path in paths:
            try:
                reports.append(self.load(path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return reports
