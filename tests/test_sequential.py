import pytest

from stoch_planning.model import PlanningConfig
from stoch_planning.sequential import StochasticProductionLookaheadPolicy
from stoch_planning.solve import solve_extensive_form


def test_capacity_decision_matches_extensive_form_root():
    policy = StochasticProductionLookaheadPolicy()
    extensive = solve_extensive_form()
    assert policy.capacity_decision() == pytest.approx(extensive.capacity_expansion)


@pytest.mark.parametrize(
    ("demand", "attribute"),
    [
        (4.0, "stage2_production_low"),
        (8.0, "stage2_production_high"),
    ],
)
def test_conditional_reoptimization_matches_extensive_form_branch_decision(
    demand,
    attribute,
):
    config = PlanningConfig()
    policy = StochasticProductionLookaheadPolicy(config)
    extensive = solve_extensive_form(config)
    rolling = policy.stage2_production_decision(
        demand,
        extensive.capacity_expansion,
    )
    assert rolling == pytest.approx(getattr(extensive, attribute), abs=1e-7)


@pytest.mark.parametrize("first_demand", [4.0, 8.0])
@pytest.mark.parametrize("second_demand", [5.0, 9.0])
def test_rolling_policy_executes_every_modeled_path(first_demand, second_demand):
    config = PlanningConfig()
    result = StochasticProductionLookaheadPolicy(config).simulate(
        first_demand,
        second_demand,
    )
    capacity = config.base_capacity + result.capacity_expansion
    assert 0 <= result.stage2_production <= capacity + 1e-8
    assert 0 <= result.stage3_production <= capacity + 1e-8
    assert result.realized_cost >= 0


def test_invalid_scenario_values_are_rejected():
    policy = StochasticProductionLookaheadPolicy()
    with pytest.raises(ValueError, match="first_demand"):
        policy.stage2_production_decision(6.0, 0.0)
    with pytest.raises(ValueError, match="second_demand"):
        policy.stage3_production_decision(0.0, 7.0, 0.0)
    with pytest.raises(ValueError, match="capacity expansion"):
        policy.stage2_production_decision(4.0, 99.0)
