from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from project_robustness_analysis import dominance, exact_sign_p, row_delta


class ProjectRobustnessTests(unittest.TestCase):
    def test_exact_sign_test(self) -> None:
        self.assertEqual(exact_sign_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_sign_p(5, 0), 0.0625)
        self.assertEqual(exact_sign_p(2, 2), 1.0)

    def test_row_delta_direction(self) -> None:
        row = {
            "line_deleted_only_ochiai_rank": 20.0,
            "line_deleted_only_ochiai_tie_expanded_size": 20,
            "line_official_omission_aware_ochiai_rank": 10.0,
            "line_official_omission_aware_ochiai_tie_expanded_size": 10,
            "line_ranking_size": 100,
            "mapped_deleted_lines": 1,
            "mapped_official_lines": 1,
        }
        delta = row_delta(row, "ochiai")
        self.assertGreater(delta["mrr_delta"], 0)
        self.assertGreater(delta["exam_improvement"], 0)
        self.assertGreater(delta["gmar_f_delta"], 0)

    def test_dominance_uses_discordant_share(self) -> None:
        projects = {
            "A": {"topk": {"10": {"gains": 3, "losses": 1, "net": 2}}},
            "B": {"topk": {"10": {"gains": 1, "losses": 1, "net": 0}}},
        }
        result = dominance(projects, 10)
        self.assertEqual(result["total_discordant"], 6)
        self.assertEqual(result["largest_discordant_project"], "A")
        self.assertAlmostEqual(result["largest_discordant_share"], 4 / 6)
        self.assertEqual(result["largest_positive_net_project"], "A")
        self.assertEqual(result["largest_positive_net_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
