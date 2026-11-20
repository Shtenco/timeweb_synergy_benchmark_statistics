import unittest

from prototypes.real_experiments.article15_measured_uncertainty import analyze_binary_benchmark


class Article15MeasuredUncertaintyTests(unittest.TestCase):
    def test_paired_seed_differences_and_wilson_intervals_are_reported(self):
        source = {
            "runs": [
                {"seed": 1, "test_rows": 10, "float32_test_accuracy": 0.9, "binary_packed_test_accuracy": 0.8},
                {"seed": 2, "test_rows": 10, "float32_test_accuracy": 0.8, "binary_packed_test_accuracy": 0.7},
                {"seed": 3, "test_rows": 10, "float32_test_accuracy": 0.9, "binary_packed_test_accuracy": 0.8},
            ]
        }

        result = analyze_binary_benchmark(source)

        self.assertEqual(result["test_rows"], 10)
        self.assertEqual(len(result["per_seed_wilson_95"]), 3)
        self.assertAlmostEqual(result["binary_minus_fp32_accuracy_mean"], -0.1)
        self.assertEqual(result["sample_size_for_5pp_margin_at_50pct"], 385)


if __name__ == "__main__":
    unittest.main()
