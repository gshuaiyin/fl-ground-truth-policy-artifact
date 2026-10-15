import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from revision_v140 import annotation_sets, matched_pools, paired_summary, random_top10_reference
from class_matched_placebo import sample_class_matched, metric_values


class RevisionTests(unittest.TestCase):
    def test_random_top10_uses_population_not_budget_as_population(self):
        self.assertAlmostEqual(random_top10_reference({'N':1000,'broad_m':1}), .01, places=9)
        self.assertGreater(random_top10_reference({'N':1000,'broad_m':16}), .14)

    def test_only_marker_anchored_relations_enter_strict_policy(self):
        p = Path(__file__).parent / 'fixtures' / 'v140'
        direct, markers, broad, strict, relations = annotation_sets(p, 'X-1')
        self.assertEqual(direct, {('A',1)})
        self.assertEqual(markers, {('A',2)})
        self.assertEqual(broad, {('A',3),('A',4)})
        self.assertEqual(strict, {('A',4)})
        self.assertEqual(len(relations), 3)

    def test_failure_matching_and_exclusion(self):
        lines = [('A',i) for i in range(7)]
        flags = np.array([True,False,True,False,True,False,True])
        pools, counts, bad = matched_pools(lines, np.array([0,1]), {0,1,6}, flags, True)
        self.assertFalse(bad)
        samples = sample_class_matched(np.random.default_rng(1), pools, counts, 100)
        self.assertTrue(all(flags[s].sum()==1 for s in samples))
        self.assertFalse(set(samples.ravel()) & {0,1,6})

    def test_insufficient_stratum_is_not_relaxed(self):
        _, _, bad = matched_pools([('A',0),('A',1)], np.array([0]), {0}, np.array([True,False]), True)
        self.assertTrue(bad)

    def test_average_topk_is_not_tie_end(self):
        obs, _ = metric_values(np.ones(3), np.array([0]), np.array([],dtype=int), np.array([[1]]))
        self.assertEqual(obs['mrr'], .5)
        self.assertEqual(obs['top1'], 0)
        self.assertEqual(obs['gmar_f'], 0)


if __name__ == '__main__':
    unittest.main()
