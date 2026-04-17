from __future__ import annotations

import math
from typing import Iterable


class BayesianDicePredictor:
    """Dirichlet-Categorical predictor for a six-sided die."""

    def __init__(self, alpha: Iterable[float] | None = None) -> None:
        prior = [1.0] * 6 if alpha is None else [float(value) for value in alpha]
        if len(prior) != 6:
            raise ValueError("alpha must contain exactly 6 values.")
        if any(value <= 0 for value in prior):
            raise ValueError("All prior parameters must be positive.")

        self._alpha = prior
        self._counts = [0] * 6

    def observe(self, outcome: int) -> None:
        """Update the posterior with one observed die roll."""
        if outcome not in {1, 2, 3, 4, 5, 6}:
            raise ValueError("Observed outcomes must be integers from 1 to 6.")
        self._counts[outcome - 1] += 1

    def observe_many(self, outcomes: Iterable[int]) -> None:
        for outcome in outcomes:
            self.observe(outcome)

    def posterior_parameters(self) -> list[float]:
        return [a + c for a, c in zip(self._alpha, self._counts)]

    def posterior_mean(self) -> list[float]:
        posterior = self.posterior_parameters()
        total = sum(posterior)
        return [value / total for value in posterior]

    def posterior_predictive(self) -> list[float]:
        return self.posterior_mean()

    def posterior_variance(self) -> list[float]:
        posterior = self.posterior_parameters()
        total = sum(posterior)
        denominator = total**2 * (total + 1)
        return [value * (total - value) / denominator for value in posterior]

    def posterior_entropy(self) -> float:
        probabilities = self.posterior_predictive()
        return -sum(p * math.log(p) for p in probabilities if p > 0)

    def uncertainty_summary(self) -> dict[str, object]:
        return {
            "probabilities": self.posterior_predictive(),
            "variances": self.posterior_variance(),
            "entropy": self.posterior_entropy(),
            "most_likely_face": self.most_likely_face(),
        }

    def most_likely_face(self) -> int:
        probabilities = self.posterior_predictive()
        return max(range(6), key=lambda index: probabilities[index]) + 1

    def counts(self) -> list[int]:
        return list(self._counts)
