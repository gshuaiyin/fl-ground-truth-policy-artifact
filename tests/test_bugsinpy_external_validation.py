from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from bugsinpy_external_validation import holm_adjust, score_policy, target_sets


class BugsInPyExternalValidationTests(unittest.TestCase):
    def test_target_sets_remove_primary_overlap(self) -> None:
        primary, extra = target_sets(
            [
                {
                    "FILE_NAME": "a.py",
                    "LINES": [4, 7],
                    "EXTENDED_LINES": [3, 4],
                }
            ]
        )
        self.assertEqual(primary, {"a.py::4", "a.py::7"})
        self.assertEqual(extra, {"a.py::3"})

    def test_score_policy_uses_average_rank_and_tie_end_for_gmar(self) -> None:
        result = score_policy(np.asarray([1.0, 0.5, 0.5, 0.0]), np.asarray([1]))
        self.assertEqual(result["rank"], 2.5)
        self.assertEqual(result["exam"], 0.625)
        self.assertEqual(result["gmar_a"], 0.25)

    def test_empty_policy_is_not_evaluable(self) -> None:
        self.assertIsNone(score_policy(np.asarray([1.0]), np.asarray([], dtype=int)))

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = holm_adjust([("a", 0.01), ("b", 0.03), ("c", 0.04)])
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)


if __name__ == "__main__":
    unittest.main()
