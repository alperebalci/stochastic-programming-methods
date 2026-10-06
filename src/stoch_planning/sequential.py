from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo

from .model import PlanningConfig
from .solve import solve_extensive_form


FIRST_STAGE_DEMANDS = (4.0, 8.0)
SECOND_STAGE_DEMANDS = (5.0, 9.0)


@dataclass(frozen=True)
class RollingPolicyResult:
    capacity_expansion: float
    first_demand: float
    stage2_production: float
    inventory_after_stage2: float
    second_demand: float
    stage3_production: float
    final_inventory: float
    realized_cost: float


def _solve_optimal(model: pyo.ConcreteModel) -> None:
    solver = pyo.SolverFactory("appsi_highs")
    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError("HiGHS solver is not available")
    results = solver.solve(model)
    term = str(results.solver.termination_condition).lower()
    if "optimal" not in term:
        raise RuntimeError(f"solver did not reach optimality: {term}")


def _inventory_penalty(net_inventory: float, config: PlanningConfig) -> float:
    if net_inventory >= 0:
        return config.holding_cost * net_inventory
    return config.shortage_cost * (-net_inventory)


@dataclass(frozen=True)
class StochasticProductionLookaheadPolicy:
    """Re-solve conditional stochastic programs as information is revealed."""

    config: PlanningConfig = PlanningConfig()
    name: str = "stochastic-DLA-production-planning"

    def __post_init__(self) -> None:
        self.config.validate()

    def capacity_decision(self) -> float:
        """Decision before either demand realization is known."""
        return solve_extensive_form(self.config).capacity_expansion

    def stage2_production_decision(
        self,
        first_demand: float,
        capacity_expansion: float,
    ) -> float:
        """Decision after first demand, while second demand remains uncertain."""
        if first_demand not in FIRST_STAGE_DEMANDS:
            raise ValueError("first_demand must be one of the modeled first-stage outcomes")
        if not 0 <= capacity_expansion <= self.config.max_capacity_expansion:
            raise ValueError("capacity expansion is outside configured bounds")

        capacity = self.config.base_capacity + capacity_expansion
        m = pyo.ConcreteModel(name="conditional_stage2_lookahead")
        m.production = pyo.Var(bounds=(0, capacity))
        m.inv1 = pyo.Var(domain=pyo.NonNegativeReals)
        m.short1 = pyo.Var(domain=pyo.NonNegativeReals)
        m.balance1 = pyo.Constraint(
            expr=m.inv1 - m.short1
            == self.config.initial_inventory + m.production - first_demand
        )

        m.production3 = pyo.Var(SECOND_STAGE_DEMANDS, bounds=(0, capacity))
        m.inv2 = pyo.Var(SECOND_STAGE_DEMANDS, domain=pyo.NonNegativeReals)
        m.short2 = pyo.Var(SECOND_STAGE_DEMANDS, domain=pyo.NonNegativeReals)
        m.balance2 = pyo.Constraint(
            SECOND_STAGE_DEMANDS,
            rule=lambda model, demand: (
                model.inv2[demand] - model.short2[demand]
                == model.inv1 - model.short1 + model.production3[demand] - demand
            ),
        )

        current_cost = (
            self.config.production_cost * m.production
            + self.config.holding_cost * m.inv1
            + self.config.shortage_cost * m.short1
        )
        future_cost = sum(
            0.5
            * (
                self.config.production_cost * m.production3[demand]
                + self.config.holding_cost * m.inv2[demand]
                + self.config.shortage_cost * m.short2[demand]
            )
            for demand in SECOND_STAGE_DEMANDS
        )
        m.objective = pyo.Objective(expr=current_cost + future_cost, sense=pyo.minimize)
        _solve_optimal(m)
        return float(pyo.value(m.production))

    def stage3_production_decision(
        self,
        current_inventory: float,
        second_demand: float,
        capacity_expansion: float,
    ) -> float:
        """Terminal recourse after the second demand is known."""
        if second_demand not in SECOND_STAGE_DEMANDS:
            raise ValueError("second_demand must be one of the modeled second-stage outcomes")
        if not 0 <= capacity_expansion <= self.config.max_capacity_expansion:
            raise ValueError("capacity expansion is outside configured bounds")

        capacity = self.config.base_capacity + capacity_expansion
        m = pyo.ConcreteModel(name="terminal_recourse")
        m.production = pyo.Var(bounds=(0, capacity))
        m.inventory = pyo.Var(domain=pyo.NonNegativeReals)
        m.shortage = pyo.Var(domain=pyo.NonNegativeReals)
        m.balance = pyo.Constraint(
            expr=m.inventory - m.shortage
            == current_inventory + m.production - second_demand
        )
        m.objective = pyo.Objective(
            expr=(
                self.config.production_cost * m.production
                + self.config.holding_cost * m.inventory
                + self.config.shortage_cost * m.shortage
            ),
            sense=pyo.minimize,
        )
        _solve_optimal(m)
        return float(pyo.value(m.production))

    def simulate(self, first_demand: float, second_demand: float) -> RollingPolicyResult:
        capacity_expansion = self.capacity_decision()
        stage2_production = self.stage2_production_decision(
            first_demand,
            capacity_expansion,
        )
        inventory_after_stage2 = (
            self.config.initial_inventory + stage2_production - first_demand
        )
        stage3_production = self.stage3_production_decision(
            inventory_after_stage2,
            second_demand,
            capacity_expansion,
        )
        final_inventory = inventory_after_stage2 + stage3_production - second_demand

        realized_cost = (
            self.config.capacity_cost * capacity_expansion
            + self.config.production_cost * stage2_production
            + _inventory_penalty(inventory_after_stage2, self.config)
            + self.config.production_cost * stage3_production
            + _inventory_penalty(final_inventory, self.config)
        )
        return RollingPolicyResult(
            capacity_expansion=capacity_expansion,
            first_demand=first_demand,
            stage2_production=stage2_production,
            inventory_after_stage2=inventory_after_stage2,
            second_demand=second_demand,
            stage3_production=stage3_production,
            final_inventory=final_inventory,
            realized_cost=realized_cost,
        )
