from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .hash import sha256_payload


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    python_version: str
    implementation: str
    platform: str
    machine: str
    package_version: str | None = None
    git_commit: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeProvenance:
        return cls(
            python_version=str(data["python_version"]),
            implementation=str(data["implementation"]),
            platform=str(data["platform"]),
            machine=str(data["machine"]),
            package_version=data.get("package_version"),
            git_commit=data.get("git_commit"),
            environment={
                str(key): str(value)
                for key, value in data.get("environment", {}).items()
            },
            tags={
                str(key): str(value)
                for key, value in data.get("tags", {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    kind: str
    identifier: str
    payload: dict[str, Any]
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        identifier: str,
        payload: dict[str, Any],
    ) -> EvidenceArtifact:
        return cls(
            kind=kind,
            identifier=identifier,
            payload=dict(payload),
            sha256=sha256_payload(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "payload": self.payload,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceArtifact:
        return cls(
            kind=str(data["kind"]),
            identifier=str(data["identifier"]),
            payload=dict(data["payload"]),
            sha256=str(data["sha256"]),
        )


@dataclass(slots=True)
class EvidenceBundle:
    autopilot_report_id: str
    artifacts: list[EvidenceArtifact]
    provenance: RuntimeProvenance
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    sha256: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "autopilot_report_id": self.autopilot_report_id,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "provenance": self.provenance.to_dict(),
        }

    def recompute_sha256(self) -> str:
        return sha256_payload(self.unsigned_dict())

    def seal(self) -> EvidenceBundle:
        self.sha256 = self.recompute_sha256()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_dict(),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceBundle:
        return cls(
            id=str(data["id"]),
            autopilot_report_id=str(data["autopilot_report_id"]),
            artifacts=[
                EvidenceArtifact.from_dict(item)
                for item in data.get("artifacts", [])
            ],
            provenance=RuntimeProvenance.from_dict(data["provenance"]),
            sha256=str(data.get("sha256", "")),
        )


@dataclass(frozen=True, slots=True)
class BundleVerification:
    valid: bool
    bundle_hash_matches: bool
    artifact_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "bundle_hash_matches": self.bundle_hash_matches,
            "artifact_failures": list(self.artifact_failures),
        }
