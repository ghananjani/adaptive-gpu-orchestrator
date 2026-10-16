from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from adaptive_gpu_orchestrator.benchmarks import (
    NCCLBaselineSpec,
    NCCLCommunicationBaseline,
)
from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    ParallelismConfig,
)


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    name: str
    group: str
    repeats: int
    available_gpus: int
    candidates: tuple[ParallelismConfig, ...]
    objective: OptimizationObjective
    constraints: OptimizationConstraints
    adapter: str
    benchmark_command: tuple[str, ...]
    timeout_s: float = 120.0
    randomize_candidates: bool = False
    random_seed: int | None = None
    nccl_baseline: NCCLBaselineSpec | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("campaign name cannot be empty")
        if not self.group.strip():
            raise ValueError("campaign group cannot be empty")
        if self.repeats <= 0:
            raise ValueError("campaign repeats must be positive")
        if self.available_gpus <= 0:
            raise ValueError("available_gpus must be positive")
        if not self.candidates:
            raise ValueError("campaign requires at least one candidate")
        if not self.benchmark_command:
            raise ValueError("campaign benchmark command cannot be empty")
        if self.timeout_s <= 0:
            raise ValueError("campaign timeout must be positive")
        if self.randomize_candidates and self.random_seed is None:
            raise ValueError(
                "random_seed is required when randomize_candidates is enabled"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "repeats": self.repeats,
            "available_gpus": self.available_gpus,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "objective": self.objective.value,
            "constraints": self.constraints.to_dict(),
            "adapter": self.adapter,
            "benchmark_command": list(self.benchmark_command),
            "timeout_s": self.timeout_s,
            "randomize_candidates": self.randomize_candidates,
            "random_seed": self.random_seed,
            "nccl_baseline": (
                self.nccl_baseline.to_dict()
                if self.nccl_baseline is not None
                else None
            ),
            "tags": dict(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignSpec:
        return cls(
            name=str(data["name"]),
            group=str(data["group"]),
            repeats=int(data["repeats"]),
            available_gpus=int(data["available_gpus"]),
            candidates=tuple(
                ParallelismConfig.from_dict(item)
                for item in data.get("candidates", [])
            ),
            objective=OptimizationObjective(data["objective"]),
            constraints=OptimizationConstraints(**data.get("constraints", {})),
            adapter=str(data.get("adapter", "generic")),
            benchmark_command=tuple(
                str(item)
                for item in data.get("benchmark_command", [])
            ),
            timeout_s=float(data.get("timeout_s", 120.0)),
            randomize_candidates=bool(
                data.get("randomize_candidates", False)
            ),
            random_seed=(
                None
                if data.get("random_seed") is None
                else int(data["random_seed"])
            ),
            nccl_baseline=(
                None
                if data.get("nccl_baseline") is None
                else NCCLBaselineSpec.from_dict(data["nccl_baseline"])
            ),
            tags={
                str(key): str(value)
                for key, value in data.get("tags", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class CampaignProvenance:
    runtime: RuntimeProvenance
    cluster_snapshot: dict[str, Any]
    package_versions: dict[str, str]
    runtime_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime.to_dict(),
            "cluster_snapshot": dict(self.cluster_snapshot),
            "package_versions": dict(self.package_versions),
            "runtime_versions": dict(self.runtime_versions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignProvenance:
        return cls(
            runtime=RuntimeProvenance.from_dict(data["runtime"]),
            cluster_snapshot=dict(data.get("cluster_snapshot", {})),
            package_versions={
                str(key): str(value)
                for key, value in data.get("package_versions", {}).items()
            },
            runtime_versions={
                str(key): str(value)
                for key, value in data.get("runtime_versions", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class CampaignRun:
    replicate: int
    status: str
    optimization_id: str | None = None
    best_config: str | None = None
    error: str | None = None
    candidate_order: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate_order"] = list(self.candidate_order)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignRun:
        return cls(
            replicate=int(data["replicate"]),
            status=str(data["status"]),
            optimization_id=data.get("optimization_id"),
            best_config=data.get("best_config"),
            error=data.get("error"),
            candidate_order=tuple(
                str(item)
                for item in data.get("candidate_order", [])
            ),
        )


@dataclass(slots=True)
class CampaignReport:
    spec: CampaignSpec
    provenance: CampaignProvenance
    runs: list[CampaignRun]
    study_id: str | None = None
    communication_baseline: NCCLCommunicationBaseline | None = None
    findings: list[str] = field(default_factory=list)
    manifest_sha256: str | None = None
    resume_count: int = 0
    resume_provenance: list[CampaignProvenance] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def completed_optimization_ids(self) -> tuple[str, ...]:
        return tuple(
            run.optimization_id
            for run in self.runs
            if run.status == "completed" and run.optimization_id is not None
        )

    @property
    def successful_replicates(self) -> int:
        return len(self.completed_optimization_ids)

    @property
    def pending_replicates(self) -> tuple[int, ...]:
        completed = {
            run.replicate
            for run in self.runs
            if run.status == "completed"
        }
        return tuple(
            replicate
            for replicate in range(1, self.spec.repeats + 1)
            if replicate not in completed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "spec": self.spec.to_dict(),
            "provenance": self.provenance.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
            "study_id": self.study_id,
            "communication_baseline": (
                self.communication_baseline.to_dict()
                if self.communication_baseline is not None
                else None
            ),
            "findings": list(self.findings),
            "manifest_sha256": self.manifest_sha256,
            "resume_count": self.resume_count,
            "resume_provenance": [
                item.to_dict()
                for item in self.resume_provenance
            ],
            "successful_replicates": self.successful_replicates,
            "pending_replicates": list(self.pending_replicates),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignReport:
        return cls(
            id=str(data["id"]),
            spec=CampaignSpec.from_dict(data["spec"]),
            provenance=CampaignProvenance.from_dict(data["provenance"]),
            runs=[
                CampaignRun.from_dict(item)
                for item in data.get("runs", [])
            ],
            study_id=data.get("study_id"),
            communication_baseline=(
                None
                if data.get("communication_baseline") is None
                else NCCLCommunicationBaseline.from_dict(
                    data["communication_baseline"]
                )
            ),
            findings=[
                str(item)
                for item in data.get("findings", [])
            ],
            manifest_sha256=data.get("manifest_sha256"),
            resume_count=int(data.get("resume_count", 0)),
            resume_provenance=[
                CampaignProvenance.from_dict(item)
                for item in data.get("resume_provenance", [])
            ],
        )
