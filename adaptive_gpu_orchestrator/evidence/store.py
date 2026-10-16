from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import EvidenceBundle


@dataclass(slots=True)
class EvidenceBundleStore:
    root: Path = Path(".ago/bundles")

    def save(self, bundle: EvidenceBundle) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{bundle.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(bundle.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, identifier: str | Path) -> EvidenceBundle:
        candidate = Path(identifier)
        path = candidate if candidate.exists() else self.root / f"{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(f"evidence bundle not found: {identifier}")
        return EvidenceBundle.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def list(self) -> list[EvidenceBundle]:
        if not self.root.exists():
            return []

        bundles: list[EvidenceBundle] = []
        paths = sorted(
            self.root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                bundles.append(self.load(path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return bundles
