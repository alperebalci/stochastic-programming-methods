import numpy as np

from sddp_storage import Cut, StorageConfig, solve_stage


def small_config() -> StorageConfig:
    return StorageConfig(
        price_supports=((30.0, 60.0), (40.0, 80.0), (35.0, 70.0)),
        probabilities=((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
        capacity=6.0,
        initial_energy=3.0,
        charge_limit=2.0,
        discharge_limit=2.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.9,
        salvage_price=30.0,
    )


def test_stage_solution_satisfies_energy_balance() -> None:
    config = small_config()
    cut = Cut(alpha=-100.0, beta=-10.0)
    result = solve_stage(config, 0, 3.0, 60.0, [cut])
    lhs = result.outgoing_energy - config.charge_efficiency * result.charge
    lhs += result.discharge / config.discharge_efficiency
    assert np.isclose(lhs, 3.0, atol=1e-8)
    assert 0 <= result.outgoing_energy <= config.capacity


def test_rhs_dual_matches_local_finite_difference() -> None:
    config = small_config()
    cut = Cut(alpha=-100.0, beta=-10.0)
    state = 3.17
    eps = 1e-5
    center = solve_stage(config, 0, state, 50.0, [cut])
    plus = solve_stage(config, 0, state + eps, 50.0, [cut])
    minus = solve_stage(config, 0, state - eps, 50.0, [cut])
    finite_difference = (plus.objective - minus.objective) / (2 * eps)
    assert np.isclose(center.incoming_state_marginal, finite_difference, atol=1e-5)
