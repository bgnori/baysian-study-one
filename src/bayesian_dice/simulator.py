from __future__ import annotations

import random
from typing import Iterable


class BiasedDie:
    """Black-box simulator for a six-sided die with configurable bias."""

    def __init__(self, probabilities: Iterable[float], seed: int | None = None) -> None:
        weights = [float(value) for value in probabilities]
        if len(weights) != 6:
            raise ValueError("A six-sided die requires exactly 6 probabilities.")
        if any(value <= 0 for value in weights):
            raise ValueError("All face probabilities must be positive.")

        total = sum(weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Probabilities must sum to 1.0.")

        self._probabilities = tuple(weights)
        self._faces = (1, 2, 3, 4, 5, 6)
        self._random = random.Random(seed)

    def roll(self) -> int:
        """Return a single die roll sampled from the hidden bias."""
        return self._random.choices(self._faces, weights=self._probabilities, k=1)[0]

    def sample(self, n: int) -> list[int]:
        """Return n observed rolls from the die."""
        if n < 0:
            raise ValueError("Sample size must be non-negative.")
        return [self.roll() for _ in range(n)]
