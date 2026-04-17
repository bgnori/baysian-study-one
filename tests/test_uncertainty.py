import unittest

from src.bayesian_dice.predictor import BayesianDicePredictor


class TestPredictorUncertainty(unittest.TestCase):
    def test_posterior_variance_decreases_after_many_observations(self):
        predictor_few = BayesianDicePredictor()
        predictor_many = BayesianDicePredictor()

        for outcome in [6, 6, 5]:
            predictor_few.observe(outcome)

        for _ in range(100):
            predictor_many.observe(6)

        variance_few = predictor_few.posterior_variance()[5]
        variance_many = predictor_many.posterior_variance()[5]

        self.assertGreater(variance_few, variance_many)

    def test_posterior_entropy_is_positive(self):
        predictor = BayesianDicePredictor()
        predictor.observe_many([6, 6, 6, 2, 6, 5])

        entropy = predictor.posterior_entropy()

        self.assertGreater(entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
