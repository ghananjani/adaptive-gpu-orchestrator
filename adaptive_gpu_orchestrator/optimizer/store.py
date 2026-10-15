from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig
from adaptive_gpu_orchestrator.optimizer.trials import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    TrialMetrics,
    TrialResult,
)


@dataclass(slots=True)
class OptimizationStore:
    root: Path = Path(".ago/optimizations")

    def save(self, result: OptimizationResult) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{result.id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def load(self, identifier: str | Path) -> OptimizationResult:
        candidate = Path(identifier)
        path = candidate if candidate.exists() else self.root / f"{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(f"optimization result not found: {identifier}")

        data = json.loads(path.read_text(encoding="utf-8"))

        def parse_trial(payload: dict | None) -> TrialResult | None:
            if payload is None:
                return None
            score = payload.get("score")
            return TrialResult(
                config=ParallelismConfig.from_dict(payload["config"]),
                metrics=TrialMetrics.from_dict(payload["metrics"]),
                feasible=bool(payload.get("feasible", True)),
                score=None if score is None else float(score),
                rejection_reasons=list(payload.get("rejection_reasons", [])),
            )

        trials: list[TrialResult] = []
        for item in data.get("trials", []):
            parsed = parse_trial(item)
            if parsed is not None:
                trials.append(parsed)

        constraints = OptimizationConstraints(**data.get("constraints", {}))
        return OptimizationResult(
            id=str(data["id"]),
            objective=OptimizationObjective(data["objective"]),
            constraints=constraints,
            trials=trials,
            best=parse_trial(data.get("best")),
            baseline=parse_trial(data.get("baseline")),
        )

    def list(self) -> list[OptimizationResult]:
        if not self.root.exists():
            return []
        paths = sorted(
            self.root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        results: list[OptimizationResult] = []
        for path in paths:
            try:
                results.append(self.load(path))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return results
