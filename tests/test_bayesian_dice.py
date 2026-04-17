import unittest

from src.bayesian_dice.predictor import BayesianDicePredictor
from src.bayesian_dice.simulator import BiasedDie


class TestBiasedDie(unittest.TestCase):
    def test_roll_returns_values_between_1_and_6(self):
        die = BiasedDie([0.05, 0.10, 0.15, 0.20, 0.20, 0.30], seed=123)

        results = [die.roll() for _ in range(200)]

        self.assertTrue(all(1 <= value <= 6 for value in results))


class TestBayesianDicePredictor(unittest.TestCase):
    def test_uniform_prior_is_used_before_any_observation(self):
        predictor = BayesianDicePredictor()

        probabilities = predictor.posterior_mean()

        self.assertEqual(len(probabilities), 6)
        for probability in probabilities:
            self.assertAlmostEqual(probability, 1 / 6, places=7)

    def test_observations_shift_posterior_toward_frequent_face(self):
        predictor = BayesianDicePredictor()
        for outcome in [6, 6, 6, 6, 2, 6, 3, 6]:
            predictor.observe(outcome)

        probabilities = predictor.posterior_mean()

        self.assertGreater(probabilities[5], probabilities[0])
        self.assertGreater(probabilities[5], probabilities[1])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=7)

    def test_predictive_distribution_matches_closed_form_update(self):
        predictor = BayesianDicePredictor(alpha=[1, 1, 1, 1, 1, 1])
        for outcome in [1, 1, 2, 6]:
            predictor.observe(outcome)

        probabilities = predictor.posterior_predictive()
        expected = [3 / 10, 2 / 10, 1 / 10, 1 / 10, 1 / 10, 2 / 10]

        for actual, target in zip(probabilities, expected):
            self.assertAlmostEqual(actual, target, places=7)


if __name__ == "__main__":
    unittest.main()
