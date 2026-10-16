from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.models import NCCLResult
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
            tags={
                str(key): str(value)
                for key, value in data.get("tags", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class CampaignNCCLBaseline:
    operation: str
    results: tuple[NCCLResult, ...]
    source: str = "nccl-tests"
    binary: str | None = None
    begin: str | None = None
    end: str | None = None
    factor: int | None = None
    gpus: int | None = None

    @property
    def summary(self) -> dict[str, float | int | None]:
        if not self.results:
            return {
                "sample_count": 0,
                "mean_algbw_gbps": None,
                "peak_algbw_gbps": None,
                "mean_busbw_gbps": None,
                "peak_busbw_gbps": None,
                "largest_size_bytes": None,
            }
        algbw = [item.algbw_gbps for item in self.results]
        busbw = [item.busbw_gbps for item in self.results]
        return {
            "sample_count": len(self.results),
            "mean_algbw_gbps": sum(algbw) / len(algbw),
            "peak_algbw_gbps": max(algbw),
            "mean_busbw_gbps": sum(busbw) / len(busbw),
            "peak_busbw_gbps": max(busbw),
            "largest_size_bytes": max(item.size_bytes for item in self.results),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "source": self.source,
            "binary": self.binary,
            "begin": self.begin,
            "end": self.end,
            "factor": self.factor,
            "gpus": self.gpus,
            "results": [item.to_dict() for item in self.results],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignNCCLBaseline:
        return cls(
            operation=str(data.get("operation", "all_reduce")),
            source=str(data.get("source", "nccl-tests")),
            binary=data.get("binary"),
            begin=data.get("begin"),
            end=data.get("end"),
            factor=(
                None if data.get("factor") is None else int(data["factor"])
            ),
            gpus=None if data.get("gpus") is None else int(data["gpus"]),
            results=tuple(
                NCCLResult(
                    operation=str(item.get("operation", "all_reduce")),
                    size_bytes=int(item["size_bytes"]),
                    time_us=float(item["time_us"]),
                    algbw_gbps=float(item["algbw_gbps"]),
                    busbw_gbps=float(item["busbw_gbps"]),
                    out_of_place=bool(item.get("out_of_place", True)),
                )
                for item in data.get("results", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class CampaignProvenance:
    runtime: RuntimeProvenance
    cluster_snapshot: dict[str, Any]
    package_versions: dict[str, str]
    cuda_toolkit_version: str | None = None
    nccl_version: str | None = None
    container_image: str | None = None
    container_image_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime.to_dict(),
            "cluster_snapshot": dict(self.cluster_snapshot),
            "package_versions": dict(self.package_versions),
            "cuda_toolkit_version": self.cuda_toolkit_version,
            "nccl_version": self.nccl_version,
            "container_image": self.container_image,
            "container_image_digest": self.container_image_digest,
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
            cuda_toolkit_version=data.get("cuda_toolkit_version"),
            nccl_version=data.get("nccl_version"),
            container_image=data.get("container_image"),
            container_image_digest=data.get("container_image_digest"),
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
    nccl_baseline: CampaignNCCLBaseline | None = None
    study_id: str | None = None
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
            "nccl_baseline": (
                self.nccl_baseline.to_dict()
                if self.nccl_baseline is not None
                else None
            ),
            "study_id": self.study_id,
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
            nccl_baseline=(
                CampaignNCCLBaseline.from_dict(data["nccl_baseline"])
                if data.get("nccl_baseline") is not None
                else None
            ),
            study_id=data.get("study_id"),
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
