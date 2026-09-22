import numpy as np
import pytest

from sddp_storage import StorageConfig, default_config


def test_default_config_is_well_formed() -> None:
    config = default_config()
    assert config.horizon == 6
    assert 0 <= config.initial_energy <= config.capacity
    for probs in config.probabilities:
        assert np.isclose(sum(probs), 1.0)


def test_invalid_probability_mass_is_rejected() -> None:
    with pytest.raises(ValueError):
        StorageConfig(price_supports=((10.0, 20.0),), probabilities=((0.2, 0.2),))
