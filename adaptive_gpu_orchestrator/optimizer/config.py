from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParallelismConfig:
    """A reproducible candidate configuration for an AI workload."""

    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    batch_size: int = 1
    concurrency: int = 1

    def __post_init__(self) -> None:
        values = (
            self.tensor_parallel,
            self.pipeline_parallel,
            self.data_parallel,
            self.batch_size,
            self.concurrency,
        )
        if any(value <= 0 for value in values):
            raise ValueError("parallelism, batch size, and concurrency must be positive")

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParallelismConfig:
        return cls(
            tensor_parallel=int(data.get("tensor_parallel", 1)),
            pipeline_parallel=int(data.get("pipeline_parallel", 1)),
            data_parallel=int(data.get("data_parallel", 1)),
            batch_size=int(data.get("batch_size", 1)),
            concurrency=int(data.get("concurrency", 1)),
        )

    def label(self) -> str:
        return (
            f"tp{self.tensor_parallel}-pp{self.pipeline_parallel}-"
            f"dp{self.data_parallel}-b{self.batch_size}-c{self.concurrency}"
        )
