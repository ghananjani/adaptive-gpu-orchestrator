from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from adaptive_gpu_orchestrator.models import NCCLResult

from .nccl import NCCLBenchmark
from .summary import summarize_nccl


@dataclass(frozen=True, slots=True)
class NCCLBaselineSpec:
    binary: str = "all_reduce_perf"
    begin: str = "8M"
    end: str = "1G"
    factor: int = 2
    gpus: int = 8
    operation: str = "all_reduce"

    def __post_init__(self) -> None:
        if not self.binary:
            raise ValueError("NCCL baseline binary cannot be empty")
        if self.factor <= 1:
            raise ValueError("NCCL baseline factor must be greater than 1")
        if self.gpus <= 0:
            raise ValueError("NCCL baseline GPU count must be positive")
        if not self.operation:
            raise ValueError("NCCL baseline operation cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary": self.binary,
            "begin": self.begin,
            "end": self.end,
            "factor": self.factor,
            "gpus": self.gpus,
            "operation": self.operation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NCCLBaselineSpec:
        return cls(
            binary=str(data.get("binary", "all_reduce_perf")),
            begin=str(data.get("begin", "8M")),
            end=str(data.get("end", "1G")),
            factor=int(data.get("factor", 2)),
            gpus=int(data.get("gpus", 8)),
            operation=str(data.get("operation", "all_reduce")),
        )


@dataclass(frozen=True, slots=True)
class NCCLCommunicationBaseline:
    spec: NCCLBaselineSpec
    results: tuple[NCCLResult, ...]
    summary: dict[str, float | int | None]
    sha256: str

    @staticmethod
    def _unsigned_payload(
        spec: NCCLBaselineSpec,
        results: tuple[NCCLResult, ...],
    ) -> dict[str, Any]:
        return {
            "spec": spec.to_dict(),
            "results": [item.to_dict() for item in results],
        }

    @staticmethod
    def _sha256(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def create(
        cls,
        spec: NCCLBaselineSpec,
        results: list[NCCLResult] | tuple[NCCLResult, ...],
    ) -> NCCLCommunicationBaseline:
        values = tuple(results)
        if not values:
            raise ValueError("NCCL baseline produced no parsed results")
        payload = cls._unsigned_payload(spec, values)
        return cls(
            spec=spec,
            results=values,
            summary=summarize_nccl(values),
            sha256=cls._sha256(payload),
        )

    def verify(self) -> bool:
        payload = self._unsigned_payload(self.spec, self.results)
        return self.sha256 == self._sha256(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_payload(self.spec, self.results),
            "summary": dict(self.summary),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NCCLCommunicationBaseline:
        spec = NCCLBaselineSpec.from_dict(data["spec"])
        results = tuple(
            NCCLResult(
                operation=str(item["operation"]),
                size_bytes=int(item["size_bytes"]),
                time_us=float(item["time_us"]),
                algbw_gbps=float(item["algbw_gbps"]),
                busbw_gbps=float(item["busbw_gbps"]),
                out_of_place=bool(item.get("out_of_place", True)),
            )
            for item in data.get("results", [])
        )
        baseline = cls(
            spec=spec,
            results=results,
            summary={
                str(key): value
                for key, value in data.get("summary", {}).items()
            },
            sha256=str(data["sha256"]),
        )
        if not baseline.verify():
            raise ValueError("NCCL communication baseline SHA-256 verification failed")
        return baseline


def capture_nccl_baseline(
    spec: NCCLBaselineSpec,
    *,
    benchmark: NCCLBenchmark | None = None,
) -> NCCLCommunicationBaseline:
    runner = benchmark or NCCLBenchmark(spec.binary)
    results = runner.run(
        begin=spec.begin,
        end=spec.end,
        factor=spec.factor,
        gpus=spec.gpus,
    )
    if spec.operation != "all_reduce":
        results = [
            NCCLResult(
                operation=spec.operation,
                size_bytes=item.size_bytes,
                time_us=item.time_us,
                algbw_gbps=item.algbw_gbps,
                busbw_gbps=item.busbw_gbps,
                out_of_place=item.out_of_place,
            )
            for item in results
        ]
    return NCCLCommunicationBaseline.create(spec, results)
