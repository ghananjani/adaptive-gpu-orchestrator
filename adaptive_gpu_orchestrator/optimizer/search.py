from __future__ import annotations

from itertools import product

from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig


def generate_candidates(
    *,
    available_gpus: int,
    tensor_parallel: list[int],
    pipeline_parallel: list[int],
    data_parallel: list[int],
    batch_sizes: list[int],
    concurrencies: list[int],
    exact_world_size: bool = False,
    max_candidates: int | None = None,
) -> list[ParallelismConfig]:
    if available_gpus <= 0:
        raise ValueError("available_gpus must be positive")

    values = (
        tensor_parallel,
        pipeline_parallel,
        data_parallel,
        batch_sizes,
        concurrencies,
    )
    if any(not items for items in values):
        raise ValueError("candidate dimensions cannot be empty")

    candidates: list[ParallelismConfig] = []
    seen: set[ParallelismConfig] = set()

    for tp, pp, dp, batch, concurrency in product(*values):
        candidate = ParallelismConfig(
            tensor_parallel=tp,
            pipeline_parallel=pp,
            data_parallel=dp,
            batch_size=batch,
            concurrency=concurrency,
        )

        if exact_world_size and candidate.world_size != available_gpus:
            continue
        if not exact_world_size and candidate.world_size > available_gpus:
            continue
        if candidate in seen:
            continue

        seen.add(candidate)
        candidates.append(candidate)
        if max_candidates is not None and len(candidates) >= max_candidates:
            break

    return candidates
