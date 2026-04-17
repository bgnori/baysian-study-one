from src.bayesian_dice.analysis import posterior_predictive_experiment
from src.bayesian_dice.simulator import BiasedDie


def main() -> None:
    die = BiasedDie([0.05, 0.10, 0.10, 0.15, 0.20, 0.40], seed=22)
    observations = die.sample(30)

    print("Advanced demo: posterior predictive simulation")
    print("observations:", observations)

    for future_samples in [20, 200]:
        summary = posterior_predictive_experiment(observations, num_samples=future_samples, seed=future_samples)
        print(f"\nFuture sample size: {future_samples}")
        print("most likely face:", summary["most_likely_face"])
        print("predictive probabilities:", [round(value, 3) for value in summary["probabilities"]])
        print("simulated counts:", summary["simulated_counts"])


if __name__ == "__main__":
    main()
