from __future__ import annotations

from dataclasses import asdict, dataclass

from .extensive import solve_extensive_form
from .model import StorageConfig, default_config
from .sddp import (
    SimulationSummary,
    TrainingResult,
    evaluate_policy_exact,
    simulate_policy,
    train_sddp,
)


@dataclass(frozen=True)
class BenchmarkResult:
    exact_objective: float
    final_lower_bound: float
    lower_bound_gap: float
    exact_policy_value: float
    exact_policy_gap: float
    policy_mean: float
    policy_standard_error: float
    policy_ci95_low: float
    policy_ci95_high: float
    policy_mean_gap: float
    iterations: int
    cuts: int
    scenario_tree_nodes: int
    scenario_tree_variables: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def run_benchmark(
    config: StorageConfig | None = None,
    iterations: int = 80,
    validation_replications: int = 2000,
    training_seed: int = 2026,
    validation_seed: int = 100_000,
) -> tuple[BenchmarkResult, TrainingResult, SimulationSummary]:
    cfg = default_config() if config is None else config
    exact = solve_extensive_form(cfg)
    training = train_sddp(cfg, iterations=iterations, seed=training_seed)
    exact_policy_value = evaluate_policy_exact(cfg, training.cuts)
    simulation = simulate_policy(
        cfg,
        training.cuts,
        replications=validation_replications,
        seed=validation_seed,
    )
    lb = training.history[-1].lower_bound
    result = BenchmarkResult(
        exact_objective=exact.objective,
        final_lower_bound=lb,
        lower_bound_gap=exact.objective - lb,
        exact_policy_value=exact_policy_value,
        exact_policy_gap=exact_policy_value - exact.objective,
        policy_mean=simulation.mean,
        policy_standard_error=simulation.standard_error,
        policy_ci95_low=simulation.ci95_low,
        policy_ci95_high=simulation.ci95_high,
        policy_mean_gap=simulation.mean - exact.objective,
        iterations=iterations,
        cuts=sum(len(stage) for stage in training.cuts),
        scenario_tree_nodes=exact.nodes,
        scenario_tree_variables=exact.variables,
    )
    return result, training, simulation
