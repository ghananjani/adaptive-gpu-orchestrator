import pytest

from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig
from adaptive_gpu_orchestrator.optimizer.search import generate_candidates


def test_parallelism_config_world_size_and_label():
    config = ParallelismConfig(
        tensor_parallel=4,
        pipeline_parallel=2,
        data_parallel=1,
        batch_size=8,
        concurrency=32,
    )
    assert config.world_size == 8
    assert config.label() == "tp4-pp2-dp1-b8-c32"


def test_parallelism_config_rejects_non_positive_values():
    with pytest.raises(ValueError):
        ParallelismConfig(tensor_parallel=0)


def test_generate_candidates_respects_gpu_budget():
    candidates = generate_candidates(
        available_gpus=8,
        tensor_parallel=[1, 2, 4, 8],
        pipeline_parallel=[1, 2],
        data_parallel=[1, 2],
        batch_sizes=[1, 4],
        concurrencies=[1],
    )
    assert candidates
    assert all(candidate.world_size <= 8 for candidate in candidates)


def test_generate_candidates_exact_world_size():
    candidates = generate_candidates(
        available_gpus=8,
        tensor_parallel=[1, 2, 4, 8],
        pipeline_parallel=[1, 2],
        data_parallel=[1, 2],
        batch_sizes=[1],
        concurrencies=[1],
        exact_world_size=True,
    )
    assert candidates
    assert all(candidate.world_size == 8 for candidate in candidates)
