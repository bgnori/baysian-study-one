from src.bayesian_dice.analysis import mean_absolute_error, posterior_trajectory, sample_posterior_predictive
from src.bayesian_dice.predictor import BayesianDicePredictor
from src.bayesian_dice.simulator import BiasedDie


def _bar(probability: float, width: int = 24) -> str:
    filled = max(1, int(round(probability * width)))
    return "#" * filled


def main() -> None:
    true_probabilities = [0.05, 0.10, 0.10, 0.15, 0.20, 0.40]
    die = BiasedDie(true_probabilities, seed=21)
    outcomes = die.sample(40)
    snapshots = posterior_trajectory(outcomes)
    final = snapshots[-1]

    print("roll\tobserved\tbest_face\tp(face 6)")
    for snapshot in snapshots[::5]:
        probabilities = snapshot["probabilities"]
        print(
            f"{snapshot['roll']}\t{snapshot['observed']}\t{snapshot['most_likely_face']}\t\t{probabilities[5]:.3f}"
        )

    final_probabilities: list[float] = final["probabilities"]
    print("\nFinal posterior predictive probabilities:")
    for face, probability in enumerate(final_probabilities, start=1):
        print(f"  face {face}: {probability:.3f}  {_bar(probability)}")

    predictor = BayesianDicePredictor()
    predictor.observe_many(outcomes)
    future_draws = sample_posterior_predictive(predictor, num_samples=20, seed=5)

    print("\nPosterior predictive sample of future rolls:")
    print(future_draws)
    print("\nMean absolute error to truth:", f"{mean_absolute_error(final_probabilities, true_probabilities):.4f}")


if __name__ == "__main__":
    main()
