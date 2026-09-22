import numpy as np

from sddp_storage import StorageConfig, solve_extensive_form


def test_one_stage_extensive_form_has_known_solution() -> None:
    config = StorageConfig(
        price_supports=((50.0,),),
        probabilities=((1.0,),),
        capacity=5.0,
        initial_energy=2.0,
        charge_limit=2.0,
        discharge_limit=2.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        salvage_price=20.0,
    )
    result = solve_extensive_form(config)
    assert np.isclose(result.objective, -100.0)
    assert result.nodes == 1
    assert result.variables == 3


def test_binary_support_tree_size() -> None:
    config = StorageConfig(
        price_supports=((20.0, 40.0),) * 3,
        probabilities=((0.5, 0.5),) * 3,
        capacity=4.0,
        initial_energy=2.0,
        charge_limit=1.0,
        discharge_limit=1.0,
        salvage_price=20.0,
    )
    result = solve_extensive_form(config)
    assert result.nodes == 2 + 4 + 8
    assert result.variables == 3 * result.nodes
