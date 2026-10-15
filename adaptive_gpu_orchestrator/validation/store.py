from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import ValidationReport


@dataclass(slots=True)
class ValidationStore:
    root: Path = Path(".ago/validations")

    def save(self, report: ValidationReport) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{report.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, identifier: str | Path) -> ValidationReport:
        candidate = Path(identifier)
        path = candidate if candidate.exists() else self.root / f"{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(f"validation report not found: {identifier}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return ValidationReport.from_dict(data)

    def list(self) -> list[ValidationReport]:
        if not self.root.exists():
            return []
        paths = sorted(
            self.root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        reports: list[ValidationReport] = []
        for path in paths:
            try:
                reports.append(self.load(path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return reports
