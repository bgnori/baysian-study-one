from src.bayesian_dice.predictor import BayesianDicePredictor
from src.bayesian_dice.simulator import BiasedDie


def main() -> None:
    die = BiasedDie([0.05, 0.10, 0.10, 0.15, 0.20, 0.40], seed=7)
    predictor = BayesianDicePredictor()

    observations = die.sample(30)
    predictor.observe_many(observations)

    print("Observed rolls:", observations)
    print("Posterior counts:", predictor.counts())
    print("Posterior predictive probabilities:")
    for face, probability in enumerate(predictor.posterior_predictive(), start=1):
        print(f"  face {face}: {probability:.3f}")
    print("Most likely next face:", predictor.most_likely_face())


if __name__ == "__main__":
    main()
