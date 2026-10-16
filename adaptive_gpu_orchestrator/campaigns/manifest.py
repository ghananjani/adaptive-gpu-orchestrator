from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_gpu_orchestrator.evidence import sha256_payload

from .models import CampaignSpec

CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    spec: CampaignSpec
    sha256: str
    schema_version: int = CAMPAIGN_MANIFEST_SCHEMA_VERSION

    @staticmethod
    def unsigned_payload(
        spec: CampaignSpec,
        schema_version: int = CAMPAIGN_MANIFEST_SCHEMA_VERSION,
    ) -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "spec": spec.to_dict(),
        }

    @classmethod
    def create(cls, spec: CampaignSpec) -> CampaignManifest:
        payload = cls.unsigned_payload(spec)
        return cls(
            spec=spec,
            sha256=sha256_payload(payload),
        )

    def verify(self) -> bool:
        return self.sha256 == sha256_payload(
            self.unsigned_payload(self.spec, self.schema_version)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_payload(self.spec, self.schema_version),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignManifest:
        schema_version = int(data.get("schema_version", 0))
        if schema_version != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported campaign manifest schema version: {schema_version}"
            )

        manifest = cls(
            spec=CampaignSpec.from_dict(data["spec"]),
            sha256=str(data["sha256"]),
            schema_version=schema_version,
        )
        if not manifest.verify():
            raise ValueError("campaign manifest SHA-256 verification failed")
        return manifest


def save_campaign_manifest(
    spec: CampaignSpec,
    path: str | Path,
) -> CampaignManifest:
    manifest = CampaignManifest.create(spec)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return manifest


def load_campaign_manifest(path: str | Path) -> CampaignManifest:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"campaign manifest not found: {path}")
    return CampaignManifest.from_dict(
        json.loads(target.read_text(encoding="utf-8"))
    )
