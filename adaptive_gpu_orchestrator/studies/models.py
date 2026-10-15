from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class MetricSummary:
    metric: str
    n: int
    mean: float
    minimum: float
    maximum: float
    stdev: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StudyGroupSummary:
    name: str
    optimization_ids: tuple[str, ...]
    metrics: dict[str, MetricSummary]
    config_counts: dict[str, int]
    dominant_config: str | None = None
    config_stability_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "optimization_ids": list(self.optimization_ids),
            "metrics": {
                name: summary.to_dict()
                for name, summary in self.metrics.items()
            },
            "config_counts": dict(self.config_counts),
            "dominant_config": self.dominant_config,
            "config_stability_fraction": self.config_stability_fraction,
        }


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric: str
    baseline_group: str
    candidate_group: str
    baseline_mean: float
    candidate_mean: float
    absolute_change: float
    percent_change: float | None = None
    relative_improvement_pct: float | None = None
    cohens_d: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    optimization_id: str
    group: str
    config_label: str
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization_id": self.optimization_id,
            "group": self.group,
            "config_label": self.config_label,
            "metrics": dict(self.metrics),
        }


@dataclass(slots=True)
class StudyReport:
    groups: list[StudyGroupSummary]
    comparisons: list[MetricComparison] = field(default_factory=list)
    pareto_frontier: list[ParetoPoint] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "groups": [group.to_dict() for group in self.groups],
            "comparisons": [
                comparison.to_dict()
                for comparison in self.comparisons
            ],
            "pareto_frontier": [
                point.to_dict()
                for point in self.pareto_frontier
            ],
            "findings": list(self.findings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudyReport:
        groups: list[StudyGroupSummary] = []
        for item in data.get("groups", []):
            groups.append(
                StudyGroupSummary(
                    name=str(item["name"]),
                    optimization_ids=tuple(
                        str(value)
                        for value in item.get("optimization_ids", [])
                    ),
                    metrics={
                        str(name): MetricSummary(**summary)
                        for name, summary in item.get("metrics", {}).items()
                    },
                    config_counts={
                        str(name): int(count)
                        for name, count in item.get("config_counts", {}).items()
                    },
                    dominant_config=item.get("dominant_config"),
                    config_stability_fraction=item.get(
                        "config_stability_fraction"
                    ),
                )
            )

        return cls(
            id=str(data["id"]),
            groups=groups,
            comparisons=[
                MetricComparison(**item)
                for item in data.get("comparisons", [])
            ],
            pareto_frontier=[
                ParetoPoint(
                    optimization_id=str(item["optimization_id"]),
                    group=str(item["group"]),
                    config_label=str(item["config_label"]),
                    metrics={
                        str(name): float(value)
                        for name, value in item.get("metrics", {}).items()
                    },
                )
                for item in data.get("pareto_frontier", [])
            ],
            findings=[str(item) for item in data.get("findings", [])],
        )
