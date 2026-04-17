from src.bayesian_dice.analysis import prior_sensitivity_report
from src.bayesian_dice.simulator import BiasedDie


def show_report(title: str, observations: list[int]) -> None:
    priors = {
        "symmetric": [1, 1, 1, 1, 1, 1],
        "face 6 favored": [1, 1, 1, 1, 1, 5],
        "strong uniform": [4, 4, 4, 4, 4, 4],
    }
    report = prior_sensitivity_report(observations, priors, num_samples=800, seed=8)

    print(f"\n[{title}]")
    print("observations:", observations)
    for name, summary in report["cases"].items():
        print(f"  - {name}")
        print(f"    most likely face: {summary['most_likely_face']}")
        print(f"    entropy estimate: {summary['entropy']:.3f}")
        print(f"    entropy width:    {summary['entropy_interval_width']:.3f}")
        print(f"    probabilities:    {[round(value, 3) for value in summary['probabilities']]}")
    print("  max probability gap across priors:", f"{report['spread']['max_probability_gap']:.3f}")


def main() -> None:
    die = BiasedDie([0.05, 0.10, 0.10, 0.15, 0.20, 0.40], seed=17)
    few_observations = die.sample(8)

    die = BiasedDie([0.05, 0.10, 0.10, 0.15, 0.20, 0.40], seed=17)
    many_observations = die.sample(60)

    print("Advanced demo: prior sensitivity analysis")
    show_report("few observations", few_observations)
    show_report("many observations", many_observations)


if __name__ == "__main__":
    main()
