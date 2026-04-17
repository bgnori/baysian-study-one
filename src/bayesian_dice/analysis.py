from __future__ import annotations

import math
import random
from typing import Iterable, TypedDict, Sequence

from .predictor import BayesianDicePredictor


class PosteriorSnapshot(TypedDict):
    roll: int
    observed: int
    counts: list[int]
    probabilities: list[float]
    most_likely_face: int


def mean_absolute_error(
    estimated: Sequence[float],
    truth: Sequence[float],
) -> float:
    """Return the mean absolute error between two discrete distributions."""
    if len(estimated) != len(truth):
        raise ValueError("Both distributions must have the same length.")
    if len(estimated) == 0:
        raise ValueError("Distributions must not be empty.")

    return sum(abs(float(a) - float(b)) for a, b in zip(estimated, truth)) / len(estimated)


def log_loss_for_observations(
    observations: Iterable[int],
    predictive_distributions: Sequence[Sequence[float]],
    epsilon: float = 1e-12,
) -> float:
    """Return the average negative log-likelihood of observed outcomes."""
    realized = list(observations)
    if len(realized) != len(predictive_distributions):
        raise ValueError("Each observation must have one predictive distribution.")
    if not realized:
        return 0.0

    total = 0.0
    for outcome, distribution in zip(realized, predictive_distributions):
        if outcome < 1 or outcome > len(distribution):
            raise ValueError("Observed outcome is outside the predictive support.")
        probability = max(float(distribution[outcome - 1]), epsilon)
        total -= math.log(probability)

    return total / len(realized)


def posterior_trajectory(
    outcomes: Iterable[int],
    alpha: Iterable[float] | None = None,
) -> list[PosteriorSnapshot]:
    """Track how posterior predictions evolve after each observation."""
    predictor = BayesianDicePredictor(alpha=alpha)
    snapshots: list[PosteriorSnapshot] = []

    for roll_index, outcome in enumerate(outcomes, start=1):
        predictor.observe(outcome)
        snapshots.append(
            {
                "roll": roll_index,
                "observed": outcome,
                "counts": predictor.counts(),
                "probabilities": predictor.posterior_predictive(),
                "most_likely_face": predictor.most_likely_face(),
            }
        )

    return snapshots


def sample_posterior_predictive(
    predictor: BayesianDicePredictor,
    num_samples: int,
    seed: int | None = None,
) -> list[int]:
    """Draw future outcomes from the current posterior predictive distribution."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")

    probabilities = predictor.posterior_predictive()
    rng = random.Random(seed)
    faces = [1, 2, 3, 4, 5, 6]
    return rng.choices(faces, weights=probabilities, k=num_samples)


def summarize_against_truth(
    outcomes: Iterable[int],
    truth: Sequence[float],
    alpha: Iterable[float] | None = None,
) -> dict[str, object]:
    """Return a compact summary of posterior convergence against the true die."""
    realized_outcomes = list(outcomes)
    snapshots = posterior_trajectory(realized_outcomes, alpha=alpha)
    if snapshots:
        final_probabilities: list[float] = snapshots[-1]["probabilities"]
        most_likely_face: int = snapshots[-1]["most_likely_face"]
    else:
        predictor = BayesianDicePredictor(alpha=alpha)
        final_probabilities = predictor.posterior_predictive()
        most_likely_face = predictor.most_likely_face()

    return {
        "num_rolls": len(realized_outcomes),
        "final_probabilities": final_probabilities,
        "most_likely_face": most_likely_face,
        "mae_to_truth": mean_absolute_error(final_probabilities, truth),
    }
