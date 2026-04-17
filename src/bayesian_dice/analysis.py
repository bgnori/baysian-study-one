from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, TypedDict, Sequence

from .predictor import BayesianDicePredictor


class PosteriorSnapshot(TypedDict):
    roll: int
    observed: int
    counts: list[int]
    probabilities: list[float]
    most_likely_face: int


class EntropyIntervalSummary(TypedDict):
    mean: float
    median: float
    lower: float
    upper: float
    width: float
    predictive_entropy: float


class EntropyTrajectorySnapshot(TypedDict):
    roll: int
    observed: int
    counts: list[int]
    probabilities: list[float]
    most_likely_face: int
    entropy_mean: float
    lower: float
    upper: float
    width: float


class PriorCaseSummary(TypedDict):
    prior: list[float]
    probabilities: list[float]
    most_likely_face: int
    entropy: float
    entropy_interval_width: float


class PriorSensitivityReport(TypedDict):
    cases: dict[str, PriorCaseSummary]
    spread: dict[str, float]


class PosteriorPredictiveExperimentSummary(TypedDict):
    probabilities: list[float]
    most_likely_face: int
    simulated_draws: list[int]
    simulated_counts: list[int]


class FairnessSummary(TypedDict):
    fair_probabilities: list[float]
    estimated_probabilities: list[float]
    mae_to_fair: float
    total_variation_to_fair: float
    max_deviation: float
    kl_to_fair: float
    most_likely_face: int


class WindowedPosteriorSnapshot(TypedDict):
    window_start: int
    window_end: int
    probabilities: list[float]
    most_likely_face: int
    total_variation_to_fair: float


class BiasChangeSummary(TypedDict):
    estimated_change_point: int
    max_shift_score: float
    windows: list[WindowedPosteriorSnapshot]


def _shannon_entropy(probabilities: Sequence[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in probabilities if p > 0)


def _sample_dirichlet(parameters: Sequence[float], rng: random.Random) -> list[float]:
    draws = [rng.gammavariate(float(value), 1.0) for value in parameters]
    total = sum(draws)
    return [draw / total for draw in draws]


def _outcome_counts(outcomes: Sequence[int]) -> list[int]:
    return [sum(1 for value in outcomes if value == face) for face in range(1, 7)]


def _total_variation_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right))


def _kl_divergence(left: Sequence[float], right: Sequence[float], epsilon: float = 1e-12) -> float:
    total = 0.0
    for left_value, right_value in zip(left, right):
        p = max(float(left_value), epsilon)
        q = max(float(right_value), epsilon)
        total += p * math.log(p / q)
    return total


def _summarize_entropy_samples(
    entropy_samples: Sequence[float],
    predictive_entropy: float,
    credible_mass: float,
) -> EntropyIntervalSummary:
    if not entropy_samples:
        raise ValueError("entropy_samples must not be empty.")
    if not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must be between 0 and 1.")

    ordered = sorted(float(value) for value in entropy_samples)
    lower_tail = (1.0 - credible_mass) / 2.0
    upper_tail = 1.0 - lower_tail
    lower_index = max(0, int(lower_tail * (len(ordered) - 1)))
    upper_index = min(len(ordered) - 1, int(upper_tail * (len(ordered) - 1)))
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[middle]
    else:
        median = 0.5 * (ordered[middle - 1] + ordered[middle])

    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return {
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "predictive_entropy": predictive_entropy,
    }


def entropy_credible_interval(
    predictor: BayesianDicePredictor,
    num_samples: int = 2000,
    credible_mass: float = 0.95,
    seed: int | None = None,
) -> EntropyIntervalSummary:
    """Estimate a credible interval for the hidden source entropy."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = random.Random(seed)
    posterior_parameters = predictor.posterior_parameters()
    entropy_samples = [
        _shannon_entropy(_sample_dirichlet(posterior_parameters, rng))
        for _ in range(num_samples)
    ]
    return _summarize_entropy_samples(
        entropy_samples,
        predictive_entropy=predictor.posterior_entropy(),
        credible_mass=credible_mass,
    )


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


def entropy_trajectory(
    outcomes: Iterable[int],
    alpha: Iterable[float] | None = None,
    num_samples: int = 1000,
    credible_mass: float = 0.95,
    seed: int | None = None,
) -> list[EntropyTrajectorySnapshot]:
    """Track posterior entropy estimates and interval width after each observation."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    predictor = BayesianDicePredictor(alpha=alpha)
    snapshots: list[EntropyTrajectorySnapshot] = []
    rng = random.Random(seed)

    for roll_index, outcome in enumerate(outcomes, start=1):
        predictor.observe(outcome)
        posterior_parameters = predictor.posterior_parameters()
        entropy_samples = [
            _shannon_entropy(_sample_dirichlet(posterior_parameters, rng))
            for _ in range(num_samples)
        ]
        summary = _summarize_entropy_samples(
            entropy_samples,
            predictive_entropy=predictor.posterior_entropy(),
            credible_mass=credible_mass,
        )
        snapshots.append(
            {
                "roll": roll_index,
                "observed": outcome,
                "counts": predictor.counts(),
                "probabilities": predictor.posterior_predictive(),
                "most_likely_face": predictor.most_likely_face(),
                "entropy_mean": summary["mean"],
                "lower": summary["lower"],
                "upper": summary["upper"],
                "width": summary["width"],
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


def posterior_predictive_experiment(
    outcomes: Iterable[int],
    alpha: Iterable[float] | None = None,
    num_samples: int = 100,
    seed: int | None = None,
) -> PosteriorPredictiveExperimentSummary:
    """Summarize a posterior predictive simulation study."""
    predictor = BayesianDicePredictor(alpha=alpha)
    predictor.observe_many(outcomes)
    simulated_draws = sample_posterior_predictive(predictor, num_samples=num_samples, seed=seed)
    return {
        "probabilities": predictor.posterior_predictive(),
        "most_likely_face": predictor.most_likely_face(),
        "simulated_draws": simulated_draws,
        "simulated_counts": _outcome_counts(simulated_draws),
    }


def prior_sensitivity_report(
    outcomes: Iterable[int],
    priors: Mapping[str, Sequence[float]],
    num_samples: int = 500,
    seed: int | None = None,
) -> PriorSensitivityReport:
    """Compare posterior summaries across multiple prior choices."""
    realized = list(outcomes)
    if not priors:
        raise ValueError("priors must not be empty.")

    cases: dict[str, PriorCaseSummary] = {}
    probability_vectors: list[list[float]] = []
    entropies: list[float] = []
    for index, (name, prior) in enumerate(priors.items()):
        predictor = BayesianDicePredictor(alpha=prior)
        predictor.observe_many(realized)
        interval = entropy_credible_interval(predictor, num_samples=num_samples, seed=None if seed is None else seed + index)
        probabilities = predictor.posterior_predictive()
        probability_vectors.append(probabilities)
        entropies.append(interval["mean"])
        cases[name] = {
            "prior": [float(value) for value in prior],
            "probabilities": probabilities,
            "most_likely_face": predictor.most_likely_face(),
            "entropy": interval["mean"],
            "entropy_interval_width": interval["width"],
        }

    max_probability_gap = 0.0
    for first in probability_vectors:
        for second in probability_vectors:
            max_probability_gap = max(
                max_probability_gap,
                max(abs(float(a) - float(b)) for a, b in zip(first, second)),
            )

    return {
        "cases": cases,
        "spread": {
            "max_probability_gap": max_probability_gap,
            "entropy_gap": max(entropies) - min(entropies),
        },
    }


def compare_to_fair_die(
    outcomes: Iterable[int],
    alpha: Iterable[float] | None = None,
) -> FairnessSummary:
    """Compare the current posterior estimate against a fair die."""
    predictor = BayesianDicePredictor(alpha=alpha)
    predictor.observe_many(outcomes)
    estimated = predictor.posterior_predictive()
    fair = [1.0 / 6.0] * 6
    return {
        "fair_probabilities": fair,
        "estimated_probabilities": estimated,
        "mae_to_fair": mean_absolute_error(estimated, fair),
        "total_variation_to_fair": _total_variation_distance(estimated, fair),
        "max_deviation": max(abs(float(a) - float(b)) for a, b in zip(estimated, fair)),
        "kl_to_fair": _kl_divergence(estimated, fair),
        "most_likely_face": predictor.most_likely_face(),
    }


def windowed_posterior_trajectory(
    outcomes: Iterable[int],
    window_size: int,
    step_size: int = 1,
    alpha: Iterable[float] | None = None,
) -> list[WindowedPosteriorSnapshot]:
    """Estimate posterior distributions on rolling windows for non-stationary detection."""
    realized = list(outcomes)
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    snapshots: list[WindowedPosteriorSnapshot] = []
    for start in range(0, max(len(realized) - window_size + 1, 0), step_size):
        window = realized[start : start + window_size]
        if len(window) < window_size:
            break
        fairness = compare_to_fair_die(window, alpha=alpha)
        snapshots.append(
            {
                "window_start": start + 1,
                "window_end": start + window_size,
                "probabilities": fairness["estimated_probabilities"],
                "most_likely_face": fairness["most_likely_face"],
                "total_variation_to_fair": fairness["total_variation_to_fair"],
            }
        )
    return snapshots


def detect_bias_change(
    outcomes: Iterable[int],
    window_size: int = 12,
    step_size: int = 1,
    alpha: Iterable[float] | None = None,
) -> BiasChangeSummary:
    """Detect the strongest change in estimated bias over time using rolling windows."""
    windows = windowed_posterior_trajectory(
        outcomes,
        window_size=window_size,
        step_size=step_size,
        alpha=alpha,
    )
    if len(windows) < 2:
        raise ValueError("At least two windows are required to detect a change.")

    best_point = windows[0]["window_end"]
    best_score = -1.0
    for previous, current in zip(windows, windows[1:]):
        score = _total_variation_distance(previous["probabilities"], current["probabilities"])
        if score > best_score:
            best_score = score
            best_point = current["window_start"]

    return {
        "estimated_change_point": best_point,
        "max_shift_score": best_score,
        "windows": windows,
    }


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
