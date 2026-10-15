from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from policy_decomposition_analysis import contrast, rank_key


def row(left: float | None, right: float | None, omission: bool = True) -> dict:
    return {
        "bug_id": "P-1",
        "has_omission_candidates": omission,
        "line_deleted_only_ochiai_rank": left,
        "line_official_direct_ochiai_rank": left,
        "line_official_omission_aware_ochiai_rank": right,
        "line_ranking_size": 100,
    }


class PolicyDecompositionTests(unittest.TestCase):
    def test_rank_key(self) -> None:
        self.assertEqual(
            rank_key("official_direct", "ochiai"),
            "line_official_direct_ochiai_rank",
        )

    def test_added_target_improves_metrics(self) -> None:
        result = contrast(
            [row(20.0, 5.0)], "official_direct", "official_omission_aware", "ochiai", 1
        )
        self.assertGreater(result["mrr"]["all_paired_mean"], 0)
        self.assertGreater(result["exam_improvement"]["all_paired_mean"], 0)
        self.assertEqual(result["rank_improvements"], 1)
        self.assertEqual(result["rank_worsenings"], 0)

    def test_evaluability_gain_tracks_right_only(self) -> None:
        result = contrast(
            [row(None, 5.0)], "official_direct", "official_omission_aware", "ochiai", 1
        )
        self.assertEqual(result["evaluable"]["net_gain"], 1)
        self.assertEqual(result["evaluable"]["right_only_bug_ids"], ["P-1"])


if __name__ == "__main__":
    unittest.main()

