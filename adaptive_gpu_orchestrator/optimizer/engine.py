from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from adaptive_gpu_orchestrator.optimizer.canary import CanaryRunner
from adaptive_gpu_orchestrator.optimizer.config import ParallelismConfig
from adaptive_gpu_orchestrator.optimizer.scoring import apply_constraints, score_trials
from adaptive_gpu_orchestrator.optimizer.trials import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationResult,
    TrialResult,
)


@dataclass(slots=True)
class ConfigurationOptimizer:
    runner: CanaryRunner
    objective: OptimizationObjective = OptimizationObjective.THROUGHPUT
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)

    def optimize(
        self,
        candidates: list[ParallelismConfig],
        *,
        baseline: ParallelismConfig | None = None,
    ) -> OptimizationResult:
        if not candidates:
            raise ValueError("at least one candidate configuration is required")

        trials: list[TrialResult] = []
        for config in candidates:
            metrics = self.runner.run(config)
            trial = TrialResult(config=config, metrics=metrics)
            trials.append(apply_constraints(trial, self.constraints))

        score_trials(trials, self.objective)
        feasible_scored = [
            trial
            for trial in trials
            if trial.feasible and trial.score is not None
        ]
        best = max(
            feasible_scored,
            key=lambda trial: (
                trial.score if trial.score is not None else float("-inf")
            ),
            default=None,
        )

        baseline_trial = None
        if baseline is not None:
            baseline_trial = next(
                (trial for trial in trials if trial.config == baseline),
                None,
            )

        return OptimizationResult(
            id=uuid4().hex[:12],
            objective=self.objective,
            constraints=self.constraints,
            trials=trials,
            best=best,
            baseline=baseline_trial,
        )
