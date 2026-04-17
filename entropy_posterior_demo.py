import math

from src.bayesian_dice.analysis import entropy_trajectory
from src.bayesian_dice.simulator import BiasedDie


def _bar(width_value: float, width: int = 24) -> str:
    filled = max(1, min(width, int(round(width_value * width))))
    return "#" * filled


def main() -> None:
    true_probabilities = [0.05, 0.10, 0.10, 0.15, 0.20, 0.40]
    true_entropy = -sum(p * math.log(p) for p in true_probabilities if p > 0)

    die = BiasedDie(true_probabilities, seed=21)
    observations = die.sample(60)
    snapshots = entropy_trajectory(observations, num_samples=1200, seed=9)

    print("Advanced demo: Bayesian estimation of source entropy")
    print("The predictor only sees observed rolls, not the hidden probabilities.")
    print("roll\tobserved\tbest_face\tmean_H\t95% CI\t\twidth")
    for snapshot in snapshots[::10]:
        print(
            f"{snapshot['roll']}\t{snapshot['observed']}\t{snapshot['most_likely_face']}\t\t"
            f"{snapshot['entropy_mean']:.3f}\t"
            f"[{snapshot['lower']:.3f}, {snapshot['upper']:.3f}]\t"
            f"{snapshot['width']:.3f}"
        )

    final = snapshots[-1]
    print("\nFinal posterior predictive probabilities:")
    for face, probability in enumerate(final["probabilities"], start=1):
        print(f"  face {face}: {probability:.3f}  {_bar(probability)}")

    print("\nEntropy summary for the hidden source:")
    print(f"  true entropy (reference only): {true_entropy:.3f}")
    print(f"  posterior mean estimate:      {final['entropy_mean']:.3f}")
    print(f"  95% credible interval:        [{final['lower']:.3f}, {final['upper']:.3f}]")
    print(f"  width shrinkage:              {snapshots[0]['width']:.3f} -> {final['width']:.3f}")


if __name__ == "__main__":
    main()
