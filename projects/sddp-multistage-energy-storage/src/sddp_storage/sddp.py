from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.stats import t as student_t

from .model import StorageConfig


@dataclass(frozen=True)
class Cut:
    """Affine lower bound alpha + beta * energy on an expected cost-to-go function."""

    alpha: float
    beta: float
    iteration: int = 0
    trial_state: float | None = None

    def value(self, energy: float) -> float:
        return self.alpha + self.beta * energy


@dataclass(frozen=True)
class StageSolution:
    objective: float
    immediate_cost: float
    charge: float
    discharge: float
    outgoing_energy: float
    incoming_state_marginal: float


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    lower_bound: float
    sampled_policy_cost: float
    cuts_added: int


@dataclass(frozen=True)
class TrainingResult:
    cuts: tuple[tuple[Cut, ...], ...]
    history: tuple[IterationRecord, ...]


@dataclass(frozen=True)
class SimulationSummary:
    mean: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    costs: np.ndarray


def _safe_initial_lower_bound(config: StorageConfig, stage: int) -> float:
    max_sale_revenue = sum(
        max(config.price_supports[t]) * config.discharge_limit
        for t in range(stage, config.horizon)
    )
    max_salvage_revenue = config.salvage_price * config.capacity
    return -(max_sale_revenue + max_salvage_revenue)


def initialize_cuts(config: StorageConfig) -> list[list[Cut]]:
    """Create valid constant lower bounds for V_t, t=1,...,T-1."""

    cuts: list[list[Cut]] = [[] for _ in range(config.horizon)]
    for stage in range(1, config.horizon):
        cuts[stage].append(Cut(alpha=_safe_initial_lower_bound(config, stage), beta=0.0))
    return cuts


def solve_stage(
    config: StorageConfig,
    stage: int,
    incoming_energy: float,
    price: float,
    future_cuts: tuple[Cut, ...] | list[Cut] | None,
) -> StageSolution:
    """Solve one observed-price stage LP and return the RHS dual for the incoming state."""

    if not 0 <= stage < config.horizon:
        raise ValueError("invalid stage")
    if incoming_energy < -1e-9 or incoming_energy > config.capacity + 1e-9:
        raise ValueError("incoming energy lies outside storage bounds")

    is_final = stage == config.horizon - 1
    n_vars = 3 if is_final else 4
    c = np.zeros(n_vars)
    c[0] = price
    c[1] = -price
    if is_final:
        c[2] = -config.salvage_price
    else:
        c[3] = 1.0
        if not future_cuts:
            raise ValueError("nonterminal stages require at least one future-value cut")

    a_eq = np.zeros((1, n_vars))
    a_eq[0, 0] = -config.charge_efficiency
    a_eq[0, 1] = 1.0 / config.discharge_efficiency
    a_eq[0, 2] = 1.0
    b_eq = np.asarray([incoming_energy], dtype=float)

    a_ub = None
    b_ub = None
    if not is_final:
        rows = []
        rhs = []
        for cut in future_cuts or ():
            row = np.zeros(n_vars)
            row[2] = cut.beta
            row[3] = -1.0
            rows.append(row)
            rhs.append(-cut.alpha)
        a_ub = np.vstack(rows)
        b_ub = np.asarray(rhs, dtype=float)

    bounds: list[tuple[float | None, float | None]] = [
        (0.0, config.charge_limit),
        (0.0, config.discharge_limit),
        (0.0, config.capacity),
    ]
    if not is_final:
        bounds.append((None, None))

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.fun is None or result.x is None:
        raise RuntimeError(f"stage LP failed: {result.message}")

    marginal = float(result.eqlin.marginals[0])
    charge, discharge, outgoing = map(float, result.x[:3])
    return StageSolution(
        objective=float(result.fun),
        immediate_cost=float(price * (charge - discharge)),
        charge=charge,
        discharge=discharge,
        outgoing_energy=outgoing,
        incoming_state_marginal=marginal,
    )


def lower_bound(
    config: StorageConfig, cuts: list[list[Cut]] | tuple[tuple[Cut, ...], ...]
) -> float:
    """Evaluate the current SDDP lower approximation at the fixed initial state."""

    values = []
    for price in config.price_supports[0]:
        future = cuts[1] if config.horizon > 1 else None
        values.append(solve_stage(config, 0, config.initial_energy, price, future).objective)
    return float(np.dot(config.probabilities[0], values))


def forward_pass(
    config: StorageConfig,
    cuts: list[list[Cut]] | tuple[tuple[Cut, ...], ...],
    rng: np.random.Generator,
) -> tuple[list[float], float]:
    """Sample one price path and simulate the current policy."""

    incoming_states: list[float] = []
    energy = config.initial_energy
    total = 0.0
    for stage in range(config.horizon):
        incoming_states.append(energy)
        probs = np.asarray(config.probabilities[stage], dtype=float)
        outcome = int(rng.choice(len(probs), p=probs))
        price = config.price_supports[stage][outcome]
        future = cuts[stage + 1] if stage < config.horizon - 1 else None
        solution = solve_stage(config, stage, energy, price, future)
        total += solution.immediate_cost
        energy = solution.outgoing_energy
    total -= config.salvage_price * energy
    return incoming_states, float(total)


def backward_pass(
    config: StorageConfig,
    cuts: list[list[Cut]],
    incoming_states: list[float],
    iteration: int,
) -> int:
    """Add one expected-value cut per noninitial stage using all price outcomes."""

    added = 0
    for stage in range(config.horizon - 1, 0, -1):
        state = incoming_states[stage]
        objectives = []
        slopes = []
        future = cuts[stage + 1] if stage < config.horizon - 1 else None
        for price in config.price_supports[stage]:
            solution = solve_stage(config, stage, state, price, future)
            objectives.append(solution.objective)
            slopes.append(solution.incoming_state_marginal)
        probs = np.asarray(config.probabilities[stage], dtype=float)
        value = float(np.dot(probs, objectives))
        beta = float(np.dot(probs, slopes))
        alpha = value - beta * state
        cuts[stage].append(Cut(alpha=alpha, beta=beta, iteration=iteration, trial_state=state))
        added += 1
    return added


def train_sddp(config: StorageConfig, iterations: int = 80, seed: int = 2026) -> TrainingResult:
    """Run single-path forward passes and full-enumeration backward passes."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    cuts = initialize_cuts(config)
    history: list[IterationRecord] = []
    previous_lb = -np.inf

    for iteration in range(1, iterations + 1):
        states, sampled_cost = forward_pass(config, cuts, rng)
        added = backward_pass(config, cuts, states, iteration)
        lb = lower_bound(config, cuts)
        if lb + 1e-7 < previous_lb:
            raise RuntimeError("SDDP lower bound decreased beyond numerical tolerance")
        previous_lb = max(previous_lb, lb)
        history.append(
            IterationRecord(
                iteration=iteration,
                lower_bound=lb,
                sampled_policy_cost=sampled_cost,
                cuts_added=added,
            )
        )

    frozen = tuple(tuple(stage_cuts) for stage_cuts in cuts)
    return TrainingResult(cuts=frozen, history=tuple(history))


def simulate_policy(
    config: StorageConfig,
    cuts: list[list[Cut]] | tuple[tuple[Cut, ...], ...],
    replications: int = 1000,
    seed: int = 100_000,
) -> SimulationSummary:
    """Estimate the expected cost of the feasible policy on independent Monte Carlo paths."""

    if replications < 2:
        raise ValueError("at least two replications are required")
    rng = np.random.default_rng(seed)
    costs = np.empty(replications, dtype=float)
    for i in range(replications):
        _, costs[i] = forward_pass(config, cuts, rng)
    mean = float(np.mean(costs))
    se = float(np.std(costs, ddof=1) / np.sqrt(replications))
    critical = float(student_t.ppf(0.975, df=replications - 1))
    half = critical * se
    return SimulationSummary(
        mean=mean,
        standard_error=se,
        ci95_low=mean - half,
        ci95_high=mean + half,
        costs=costs,
    )


def evaluate_policy_exact(
    config: StorageConfig,
    cuts: list[list[Cut]] | tuple[tuple[Cut, ...], ...],
) -> float:
    """Enumerate the finite price tree to compute the exact expectation of the induced policy."""

    def recurse(stage: int, energy: float) -> float:
        expected = 0.0
        for price, probability in zip(
            config.price_supports[stage], config.probabilities[stage], strict=True
        ):
            future = cuts[stage + 1] if stage < config.horizon - 1 else None
            solution = solve_stage(config, stage, energy, price, future)
            value = solution.immediate_cost
            if stage == config.horizon - 1:
                value -= config.salvage_price * solution.outgoing_energy
            else:
                value += recurse(stage + 1, solution.outgoing_energy)
            expected += probability * value
        return float(expected)

    return recurse(0, config.initial_energy)
