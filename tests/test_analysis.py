import math
import unittest

from src.bayesian_dice.analysis import (
    log_loss_for_observations,
    mean_absolute_error,
    posterior_trajectory,
)


class TestAnalysisUtilities(unittest.TestCase):
    def test_mean_absolute_error_is_zero_for_identical_distributions(self):
        distribution = [1 / 6] * 6

        error = mean_absolute_error(distribution, distribution)

        self.assertAlmostEqual(error, 0.0, places=10)

    def test_posterior_trajectory_reflects_repeated_evidence(self):
        snapshots = posterior_trajectory([6, 6, 6, 6, 2, 6])

        self.assertEqual(len(snapshots), 6)
        self.assertEqual(snapshots[-1]["most_likely_face"], 6)
        self.assertGreater(
            snapshots[-1]["probabilities"][5],
            snapshots[0]["probabilities"][5],
        )

    def test_log_loss_matches_manual_value(self):
        observations = [1, 2]
        predictive_distributions = [
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
        ]

        loss = log_loss_for_observations(observations, predictive_distributions)
        expected = -0.5 * (math.log(0.5) + math.log(0.75))

        self.assertAlmostEqual(loss, expected, places=10)


if __name__ == "__main__":
    unittest.main()
