import unittest

from llmao_external_validation import official_order_rank, ranks_desc


class LLMAOValidationTests(unittest.TestCase):
    def test_numeric_ranking_uses_average_and_end_ranks_for_ties(self):
        average, end = ranks_desc({10: 0.9, 20: 0.5, 30: 0.5, 40: 0.1})
        self.assertEqual(average, {10: 1.0, 20: 2.5, 30: 2.5, 40: 4.0})
        self.assertEqual(end, {10: 1, 20: 3, 30: 3, 40: 4})


    def test_numeric_ranking_does_not_depend_on_json_order(self):
        first, _ = ranks_desc({20: 0.2, 10: 0.9})
        second, _ = ranks_desc({10: 0.9, 20: 0.2})
        self.assertEqual(first, second)
        self.assertEqual(first, {10: 1.0, 20: 2.0})


    def test_official_reproduction_preserves_score_collision_behavior(self):
        raw = {"10": 0.8, "20": 0.8, "30": 0.1}
        self.assertIsNone(official_order_rank(raw, {10}))
        self.assertEqual(official_order_rank(raw, {20}), 1)


if __name__ == "__main__":
    unittest.main()
