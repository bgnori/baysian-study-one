import unittest

from src.bayesian_dice.analysis import sample_posterior_predictive
from src.bayesian_dice.predictor import BayesianDicePredictor


class TestPosteriorPredictive(unittest.TestCase):
    def test_sample_posterior_predictive_returns_valid_faces(self):
        predictor = BayesianDicePredictor()
        predictor.observe_many([6, 6, 5, 6, 4, 6])

        draws = sample_posterior_predictive(predictor, num_samples=50, seed=123)

        self.assertEqual(len(draws), 50)
        self.assertTrue(all(1 <= draw <= 6 for draw in draws))

    def test_favored_face_appears_more_often_in_predictive_sampling(self):
        predictor = BayesianDicePredictor()
        predictor.observe_many([6] * 40 + [5] * 5)

        draws = sample_posterior_predictive(predictor, num_samples=300, seed=7)

        self.assertGreater(draws.count(6), draws.count(1))


if __name__ == "__main__":
    unittest.main()
