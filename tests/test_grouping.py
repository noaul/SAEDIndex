import unittest
from pathlib import Path

import numpy as np

from fixtures.pattern_cases import cases, image_case, make_case, truth_match
from saedindex.pattern import detect_spots, find_center, find_lattices


class GroupingTests(unittest.TestCase):
    def test_known_group_counts(self):
        for case in cases():
            with self.subTest(case=case.name):
                groups = find_lattices(case.spots, case.center, max_sets=3)
                self.assertEqual(len(groups), case.expected_groups)
                matched = set()
                for group in groups:
                    matches = truth_match(case, group)
                    best = max(range(len(matches)), key=lambda i: sum(matches[i]))
                    precision, recall = matches[best]
                    self.assertGreaterEqual(precision, .85)
                    self.assertGreaterEqual(recall, .85)
                    matched.add(best)
                self.assertEqual(len(matched), case.expected_groups)

    def test_repeatable_and_respects_explicit_group_limit(self):
        case = next(c for c in cases() if c.name == "double_rotated")
        first = find_lattices(case.spots, case.center, max_sets=1)
        second = find_lattices(case.spots, case.center, max_sets=1)
        self.assertEqual(len(first), 1)
        np.testing.assert_allclose(first[0].g1, second[0].g1)
        self.assertEqual(first[0].members, second[0].members)

    def test_saturated_center_detection_preserves_single_group(self):
        case, image = image_case()
        spots = detect_spots(image)
        center = find_center(image, spots)
        np.testing.assert_allclose(center, case.center, atol=1.)
        self.assertEqual(len(find_lattices(spots, center)), 1)

    def test_obscured_center_can_be_manually_supplied(self):
        case, image = image_case(obscured=True)
        spots = detect_spots(image)
        self.assertEqual(len(find_lattices(spots, case.center)), 1)

    def test_low_occupancy_mesh_does_not_discard_other_supported_hypotheses(self):
        from saedindex.session import Session
        path = Path(r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000\1-0009.jpg")
        if not path.exists():
            self.skipTest("Local regression image unavailable")
        session = Session()
        session.open(path)
        session.detect()
        groups = find_lattices(session.spots, session.center)
        self.assertTrue(any(len(g.members) >= 10 for g in groups))

    def test_six_accidental_noise_nodes_need_independent_symmetry_evidence(self):
        case = make_case("accidental_six", [], seed=26, noise=65)
        report = []
        self.assertEqual(find_lattices(case.spots, case.center, report=report), [])
        self.assertTrue(any(not r["accepted"] for r in report))


if __name__ == "__main__":
    unittest.main()
