from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linprog

from .model import StorageConfig


@dataclass(frozen=True)
class ExtensiveFormResult:
    objective: float
    nodes: int
    variables: int


def solve_extensive_form(
    config: StorageConfig,
    initial_energy: float | None = None,
    start_stage: int = 0,
) -> ExtensiveFormResult:
    """Solve the full scenario-tree deterministic equivalent LP for verification."""

    if not 0 <= start_stage < config.horizon:
        raise ValueError("invalid start stage")
    energy0 = config.initial_energy if initial_energy is None else float(initial_energy)
    if not 0 <= energy0 <= config.capacity:
        raise ValueError("initial energy lies outside capacity")

    histories_by_stage: dict[int, list[tuple[int, ...]]] = {}
    all_nodes: list[tuple[int, tuple[int, ...]]] = []
    for stage in range(start_stage, config.horizon):
        supports = [range(len(config.price_supports[t])) for t in range(start_stage, stage + 1)]
        histories = list(product(*supports))
        histories_by_stage[stage] = histories
        all_nodes.extend((stage, history) for history in histories)

    node_index = {node: i for i, node in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    n_vars = 3 * n_nodes
    c = np.zeros(n_vars)
    a_eq = np.zeros((n_nodes, n_vars))
    b_eq = np.zeros(n_nodes)
    bounds: list[tuple[float, float]] = []
    for _ in range(n_nodes):
        bounds.extend(
            [
                (0.0, config.charge_limit),
                (0.0, config.discharge_limit),
                (0.0, config.capacity),
            ]
        )

    for row, (stage, history) in enumerate(all_nodes):
        idx = node_index[(stage, history)]
        base = 3 * idx
        current_outcome = history[-1]
        probability = 1.0
        for offset, outcome in enumerate(history):
            probability *= config.probabilities[start_stage + offset][outcome]
        price = config.price_supports[stage][current_outcome]

        c[base] = probability * price
        c[base + 1] = -probability * price
        if stage == config.horizon - 1:
            c[base + 2] += -probability * config.salvage_price

        a_eq[row, base] = -config.charge_efficiency
        a_eq[row, base + 1] = 1.0 / config.discharge_efficiency
        a_eq[row, base + 2] = 1.0
        if stage == start_stage:
            b_eq[row] = energy0
        else:
            parent = (stage - 1, history[:-1])
            parent_base = 3 * node_index[parent]
            a_eq[row, parent_base + 2] = -1.0

    result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success or result.fun is None:
        raise RuntimeError(f"extensive-form LP failed: {result.message}")
    return ExtensiveFormResult(objective=float(result.fun), nodes=n_nodes, variables=n_vars)
