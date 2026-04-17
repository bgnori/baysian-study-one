from src.bayesian_dice.predictor import BayesianDicePredictor
from src.bayesian_dice.simulator import BiasedDie


def run_case(name: str, alpha: list[float], observations: list[int]) -> None:
    predictor = BayesianDicePredictor(alpha=alpha)
    predictor.observe_many(observations)
    summary = predictor.uncertainty_summary()

    print(f"\n[{name}]")
    print("prior alpha:", alpha)
    print("most likely face:", summary["most_likely_face"])
    print("entropy:", f"{summary['entropy']:.4f}")
    print("probabilities:")
    for face, probability in enumerate(summary["probabilities"], start=1):
        print(f"  face {face}: {probability:.3f}")


def main() -> None:
    die = BiasedDie([0.05, 0.10, 0.10, 0.15, 0.20, 0.40], seed=99)
    observations = die.sample(12)

    print("observations:", observations)
    run_case("symmetric prior", [1, 1, 1, 1, 1, 1], observations)
    run_case("face 6 favored prior", [1, 1, 1, 1, 1, 4], observations)
    run_case("stronger uniform prior", [3, 3, 3, 3, 3, 3], observations)


if __name__ == "__main__":
    main()
