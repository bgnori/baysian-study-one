from src.bayesian_dice.analysis import compare_to_fair_die
from src.bayesian_dice.simulator import BiasedDie


def show_case(name: str, observations: list[int]) -> None:
    summary = compare_to_fair_die(observations)
    print(f"\n[{name}]")
    print("most likely face:", summary["most_likely_face"])
    print("estimated probabilities:", [round(value, 3) for value in summary["estimated_probabilities"]])
    print("MAE to fair die:", f"{summary['mae_to_fair']:.4f}")
    print("TVD to fair die:", f"{summary['total_variation_to_fair']:.4f}")
    print("KL to fair die:", f"{summary['kl_to_fair']:.4f}")


def main() -> None:
    fair_die = BiasedDie([1 / 6] * 6, seed=31)
    biased_die = BiasedDie([0.05, 0.10, 0.10, 0.15, 0.20, 0.40], seed=31)

    print("Advanced demo: comparison with a fair die")
    show_case("fair source", fair_die.sample(60))
    show_case("biased source", biased_die.sample(60))


if __name__ == "__main__":
    main()
