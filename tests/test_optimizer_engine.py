from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig
from adaptive_gpu_orchestrator.optimizer.engine import ConfigurationOptimizer
from adaptive_gpu_orchestrator.optimizer.store import OptimizationStore
from adaptive_gpu_orchestrator.optimizer.trials import (
    OptimizationConstraints,
    OptimizationObjective,
    TrialMetrics,
)


class FakeRunner:
    def run(self, config):
        return TrialMetrics(
            throughput_per_s=float(config.batch_size * config.tensor_parallel * 100),
            p99_latency_ms=float(200 / config.tensor_parallel),
            energy_joules=float(1000 + config.batch_size * 10),
            memory_pressure_pct=float(min(99, 40 + config.batch_size * 5)),
            exit_code=0,
        )


def test_optimizer_selects_best_feasible_candidate():
    candidates = [
        ParallelismConfig(tensor_parallel=1, batch_size=1),
        ParallelismConfig(tensor_parallel=2, batch_size=2),
        ParallelismConfig(tensor_parallel=4, batch_size=4),
    ]
    optimizer = ConfigurationOptimizer(
        runner=FakeRunner(),
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(max_memory_pressure_pct=70),
    )

    result = optimizer.optimize(candidates, baseline=candidates[0])

    assert result.best is not None
    assert result.best.config.tensor_parallel == 4
    assert result.baseline is not None
    assert result.baseline.config == candidates[0]


def test_optimization_store_round_trip(tmp_path):
    candidate = ParallelismConfig(tensor_parallel=2, batch_size=2)
    result = ConfigurationOptimizer(
        runner=FakeRunner(),
        objective=OptimizationObjective.BALANCED,
    ).optimize([candidate], baseline=candidate)

    store = OptimizationStore(tmp_path)
    path = store.save(result)
    loaded = store.load(result.id)

    assert path.exists()
    assert loaded.id == result.id
    assert loaded.objective is OptimizationObjective.BALANCED
    assert loaded.best is not None
    assert loaded.best.config == candidate
