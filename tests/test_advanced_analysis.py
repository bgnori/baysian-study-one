import unittest

from src.bayesian_dice.analysis import (
    compare_to_fair_die,
    detect_bias_change,
    posterior_predictive_experiment,
    prior_sensitivity_report,
)


class TestAdvancedBayesianAnalysis(unittest.TestCase):
    def test_prior_sensitivity_gap_shrinks_with_more_observations(self):
        priors = {
            "uniform": [1, 1, 1, 1, 1, 1],
            "face6_favored": [1, 1, 1, 1, 1, 6],
        }

        few = prior_sensitivity_report([6, 6, 5, 6], priors, num_samples=250, seed=5)
        many = prior_sensitivity_report([6] * 80 + [5] * 10 + [4] * 10, priors, num_samples=250, seed=5)

        self.assertGreater(few["spread"]["max_probability_gap"], many["spread"]["max_probability_gap"])

    def test_posterior_predictive_experiment_returns_valid_counts(self):
        summary = posterior_predictive_experiment([6] * 20 + [5] * 5, num_samples=80, seed=12)

        self.assertEqual(sum(summary["simulated_counts"]), 80)
        self.assertGreater(summary["simulated_counts"][5], summary["simulated_counts"][0])

    def test_fairness_comparison_distinguishes_fair_and_biased_rolls(self):
        fair_summary = compare_to_fair_die([1, 2, 3, 4, 5, 6] * 8)
        biased_summary = compare_to_fair_die([6] * 30 + [5] * 10 + [4] * 8)

        self.assertLess(fair_summary["mae_to_fair"], biased_summary["mae_to_fair"])
        self.assertLess(fair_summary["total_variation_to_fair"], biased_summary["total_variation_to_fair"])

    def test_bias_change_detection_finds_transition_region(self):
        observations = [1] * 25 + [6] * 25

        summary = detect_bias_change(observations, window_size=10, step_size=5)

        self.assertGreaterEqual(summary["estimated_change_point"], 20)
        self.assertLessEqual(summary["estimated_change_point"], 30)
        self.assertGreater(summary["max_shift_score"], 0.2)


if __name__ == "__main__":
    unittest.main()
