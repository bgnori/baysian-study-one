import math
import unittest

from src.bayesian_dice.analysis import entropy_credible_interval, entropy_trajectory
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

    def test_entropy_credible_interval_stays_within_theoretical_bounds(self):
        predictor = BayesianDicePredictor()
        predictor.observe_many([6, 6, 6, 5, 6, 4, 6, 6])

        interval = entropy_credible_interval(predictor, num_samples=400, seed=11)

        self.assertGreaterEqual(interval["lower"], 0.0)
        self.assertLessEqual(interval["upper"], math.log(6))
        self.assertGreater(interval["width"], 0.0)

    def test_entropy_interval_width_shrinks_with_more_observations(self):
        early = entropy_trajectory([6, 6, 5, 6, 4], num_samples=300, seed=7)
        late = entropy_trajectory([6] * 80 + [5] * 10 + [4] * 10, num_samples=300, seed=7)

        self.assertGreater(early[-1]["width"], late[-1]["width"])


if __name__ == "__main__":
    unittest.main()
