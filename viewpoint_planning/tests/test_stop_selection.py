"""Coverage targets must be attainable and survive stop pruning."""
import unittest
from unittest.mock import patch

import numpy as np

from candidate_solution import solution


class StopSelectionTests(unittest.TestCase):
    def select(self, sets, total=1000, target=0.95):
        arrays = [np.array(cells, dtype=np.int32) for cells in sets]
        with patch.object(solution, 'COVERAGE_TARGET', target):
            return solution._greedy_set_cover(arrays, total)

    def test_target_uses_candidate_union_not_global_denominator(self):
        # Nine cells meet 90% of ten attainable cells, despite 1000 total walls.
        sets = [list(range(9)), [9]]
        self.assertEqual(self.select(sets, target=0.9), [0])

    def test_small_gains_and_more_than_forty_stops_are_allowed(self):
        sets = [[i] for i in range(50)]
        self.assertEqual(self.select(sets, target=1), list(range(50)))

    def test_prunes_an_earlier_stop_made_redundant_by_later_stops(self):
        sets = [[0, 1, 2, 3], [0, 1, 4], [2, 3, 5]]
        self.assertEqual(self.select(sets, target=1), [1, 2])

    def test_empty_union(self):
        self.assertEqual(self.select([]), [])
        self.assertEqual(self.select([[], []], total=0), [])

    def test_duplicate_cells_do_not_inflate_gain(self):
        self.assertEqual(self.select([[0, 0, 0], [0, 1]], target=1), [1])

    def test_targets_survive_pruning_and_result_is_deterministic(self):
        rng = np.random.default_rng(17)
        sets = [np.flatnonzero(rng.random(80) < 0.2).tolist() for _ in range(35)]
        union = set().union(*map(set, sets))
        for target in [0.5, 0.9, 0.95, 0.995, 1]:
            with self.subTest(target=target):
                chosen = self.select(sets, target=target)
                self.assertEqual(chosen, self.select(sets, target=target))
                required = int(np.ceil(target * len(union)))
                covered = set().union(*(sets[i] for i in chosen))
                self.assertGreaterEqual(len(covered), required)
                # No individual selected stop can now be removed at this target.
                for removed in chosen:
                    remaining = set().union(*(sets[i] for i in chosen if i != removed))
                    self.assertLess(len(remaining), required)


if __name__ == '__main__':
    unittest.main()
