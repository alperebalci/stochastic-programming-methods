import numpy as np

from sddp_storage import (
    StorageConfig,
    evaluate_policy_exact,
    solve_extensive_form,
    train_sddp,
)


def compact_config() -> StorageConfig:
    return StorageConfig(
        price_supports=(
            (25.0, 55.0),
            (20.0, 70.0),
            (35.0, 80.0),
            (30.0, 65.0),
        ),
        probabilities=((0.5, 0.5),) * 4,
        capacity=6.0,
        initial_energy=3.0,
        charge_limit=2.0,
        discharge_limit=2.0,
        charge_efficiency=0.92,
        discharge_efficiency=0.90,
        salvage_price=30.0,
    )


def test_lower_bound_is_monotone_and_below_extensive_optimum() -> None:
    config = compact_config()
    exact = solve_extensive_form(config).objective
    training = train_sddp(config, iterations=20, seed=11)
    bounds = np.asarray([record.lower_bound for record in training.history])
    assert np.all(np.diff(bounds) >= -1e-7)
    assert np.all(bounds <= exact + 1e-6)


def test_generated_cuts_are_valid_on_energy_grid() -> None:
    config = compact_config()
    training = train_sddp(config, iterations=18, seed=7)
    for stage in range(1, config.horizon):
        for energy in np.linspace(0.0, config.capacity, 7):
            exact = solve_extensive_form(
                config, initial_energy=float(energy), start_stage=stage
            ).objective
            approximation = max(cut.value(float(energy)) for cut in training.cuts[stage])
            assert approximation <= exact + 2e-6


def test_exact_policy_expectation_is_a_valid_upper_bound() -> None:
    config = compact_config()
    exact = solve_extensive_form(config).objective
    training = train_sddp(config, iterations=25, seed=17)
    policy_value = evaluate_policy_exact(config, training.cuts)
    assert policy_value >= exact - 1e-6
    assert policy_value - exact < 1.0


def test_training_is_reproducible() -> None:
    config = compact_config()
    a = train_sddp(config, iterations=8, seed=123)
    b = train_sddp(config, iterations=8, seed=123)
    assert [r.lower_bound for r in a.history] == [r.lower_bound for r in b.history]
