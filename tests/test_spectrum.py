import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from spectrum import (
    average_ranks_desc,
    deleted_source_locations,
    fractional_top_k_weights,
    suspiciousness,
    weighted_jaccard,
)
from tie_breaker_gate import lexicographic_average_ranks_desc


class SpectrumTests(unittest.TestCase):
    def test_average_ranks_desc_with_ties(self):
        actual = average_ranks_desc(np.asarray([0.5, 0.8, 0.5, 0.1]))
        np.testing.assert_allclose(actual, [2.5, 1.0, 2.5, 4.0])

    def test_ochiai(self):
        coverage = np.asarray([[1, 1], [1, 0], [0, 1]], dtype=bool)
        failed = np.asarray([True, False, False])
        actual = suspiciousness(coverage, failed, "ochiai")
        np.testing.assert_allclose(actual, [1 / np.sqrt(2), 1 / np.sqrt(2)])

    def test_dstar_infinite_case_uses_finite_order_preserving_cap(self):
        coverage = np.asarray([[1, 1], [0, 1]], dtype=bool)
        failed = np.asarray([True, False])
        actual = suspiciousness(coverage, failed, "dstar")
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(actual[0], actual[1])

    def test_chart_source_patch_location(self):
        path = Path(__file__).parent / "fixtures" / "chart.src.patch"
        self.assertEqual(deleted_source_locations(path), {"org.example.A#10"})

    def test_fractional_top_k_weights_tie_boundary(self):
        weights = fractional_top_k_weights(np.asarray([1.0, 0.5, 0.5, 0.5, 0.0]), 2)
        np.testing.assert_allclose(weights, [1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
        self.assertAlmostEqual(weights.sum(), 2.0)
        self.assertAlmostEqual(weighted_jaccard(weights, weights), 1.0)

    def test_lexicographic_secondary_breaks_only_primary_ties(self):
        primary = np.asarray([1.0, 1.0, 0.5, 0.5])
        secondary = np.asarray([0.2, 0.8, 0.9, 0.9])
        actual = lexicographic_average_ranks_desc(primary, secondary)
        np.testing.assert_allclose(actual, [2.0, 1.0, 3.5, 3.5])


if __name__ == "__main__":
    unittest.main()
