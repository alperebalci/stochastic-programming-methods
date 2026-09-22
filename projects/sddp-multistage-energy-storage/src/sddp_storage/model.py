from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StorageConfig:
    """Finite-horizon linear battery arbitrage model with stagewise-independent prices."""

    price_supports: tuple[tuple[float, ...], ...]
    probabilities: tuple[tuple[float, ...], ...]
    capacity: float = 10.0
    initial_energy: float = 5.0
    charge_limit: float = 4.0
    discharge_limit: float = 4.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.90
    salvage_price: float = 35.0

    def __post_init__(self) -> None:
        if len(self.price_supports) == 0:
            raise ValueError("at least one stage is required")
        if len(self.price_supports) != len(self.probabilities):
            raise ValueError("price supports and probabilities must have equal length")
        if self.capacity <= 0 or self.charge_limit < 0 or self.discharge_limit < 0:
            raise ValueError("invalid storage limits")
        if not 0 <= self.initial_energy <= self.capacity:
            raise ValueError("initial energy must lie within storage capacity")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("efficiencies must lie in (0, 1]")
        if self.salvage_price < 0:
            raise ValueError("salvage price must be nonnegative")
        for support, probs in zip(self.price_supports, self.probabilities, strict=True):
            if len(support) == 0 or len(support) != len(probs):
                raise ValueError("every stage needs matching nonempty support and probabilities")
            if any(price <= 0 for price in support):
                raise ValueError("this teaching model requires strictly positive prices")
            if any(prob < 0 for prob in probs) or not np.isclose(sum(probs), 1.0):
                raise ValueError("stage probabilities must be nonnegative and sum to one")

    @property
    def horizon(self) -> int:
        return len(self.price_supports)


def default_config() -> StorageConfig:
    """Return a small seasonal instance suitable for exact extensive-form verification."""

    return StorageConfig(
        price_supports=(
            (28.0, 42.0, 58.0),
            (24.0, 40.0, 66.0),
            (20.0, 46.0, 82.0),
            (34.0, 57.0, 92.0),
            (30.0, 51.0, 76.0),
            (26.0, 44.0, 68.0),
        ),
        probabilities=(
            (0.25, 0.50, 0.25),
            (0.30, 0.45, 0.25),
            (0.25, 0.45, 0.30),
            (0.20, 0.45, 0.35),
            (0.25, 0.50, 0.25),
            (0.30, 0.45, 0.25),
        ),
        capacity=10.0,
        initial_energy=5.0,
        charge_limit=4.0,
        discharge_limit=4.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.90,
        salvage_price=35.0,
    )
