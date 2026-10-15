from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from class_matched_placebo import sample_class_matched, tie_inspection_sizes


class ClassMatchedPlaceboTests(unittest.TestCase):
    def test_sampling_preserves_class_counts_and_pool_membership(self) -> None:
        pools = {
            "A": np.asarray([1, 2, 3]),
            "B": np.asarray([10, 11, 12, 13]),
        }
        samples = sample_class_matched(
            np.random.default_rng(7), pools, {"A": 2, "B": 1}, 100
        )
        self.assertEqual(samples.shape, (100, 3))
        for sample in samples:
            self.assertEqual(len(set(sample[:2])), 2)
            self.assertTrue(set(sample[:2]).issubset(set(pools["A"])))
            self.assertIn(sample[2], pools["B"])

    def test_tie_inspection_expands_to_end_of_tie_group(self) -> None:
        scores = np.asarray([0.8, 0.5, 0.8, 0.2, 0.5])
        np.testing.assert_array_equal(
            tie_inspection_sizes(scores), np.asarray([2, 4, 2, 5, 4])
        )

    def test_sampling_rejects_insufficient_pool(self) -> None:
        with self.assertRaises(ValueError):
            sample_class_matched(
                np.random.default_rng(7), {"A": np.asarray([1])}, {"A": 2}, 10
            )


if __name__ == "__main__":
    unittest.main()
