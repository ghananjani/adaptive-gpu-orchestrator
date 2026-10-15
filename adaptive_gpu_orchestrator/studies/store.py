from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import StudyReport


@dataclass(slots=True)
class StudyStore:
    root: Path = Path(".ago/studies")

    def save(self, report: StudyReport) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{report.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, identifier: str | Path) -> StudyReport:
        candidate = Path(identifier)
        path = candidate if candidate.exists() else self.root / f"{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(f"study report not found: {identifier}")
        return StudyReport.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def list(self) -> list[StudyReport]:
        if not self.root.exists():
            return []

        reports: list[StudyReport] = []
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
