import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from saedindex.session import Session
from saedindex.rendering import FigureStyle


class ProjectTests(unittest.TestCase):
    def test_cross_drive_reference_can_be_saved_and_resolved(self):
        from saedindex.project import _reference, _resolve
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.txt"
            source.write_text("source", encoding="utf-8")
            root = Path(__file__).resolve().parent
            if source.drive.lower() == root.drive.lower():
                self.skipTest("Cross-drive test requires two Windows volumes")
            ref = _reference(source, root)
            self.assertTrue(Path(ref["path"]).is_absolute())
            self.assertEqual(_resolve(ref, root), source.resolve())

    def test_round_trip_retains_geometry_confirmation_and_style(self):
        from saedindex.project import save_project, load_project
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "pattern.png"
            Image.fromarray(np.zeros((512, 512), np.uint8)).save(image)
            cif = root / "simple.cif"
            cif.write_text("""data_test
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ni 0 0 0
""", encoding="utf-8")
            s = Session()
            s.open(image)
            s.add_phase(cif)
            s.px_per_unit = 40.
            s.image.calibration.px_per_unit = 40.
            s.image.calibration.confident = True
            s.center = np.array([256., 256.])
            from saedindex.pattern import Spot
            s.spots = [Spot(356, 256, 100), Spot(256, 156, 100), Spot(356, 156, 100)]
            group = s.set_from_spots(0, 1)
            s.apply_reference([[20, 460], [220, 460]], 5., "1/nm")
            group.confirmed = True
            group.label_offsets["g1"] = (30., -10.)
            group.label_offsets["zone"] = (-40., -20.)
            group.color = "#ffd400"
            style = FigureStyle(gamma=1.3, font_size=32, font_bold=True, line_width=5,
                                cover_rect=[10, 400, 240, 490], simulation_size=9.)
            path = root / "pattern.saed.json"
            save_project(path, s, style)
            loaded, loaded_style = load_project(path)
            self.assertEqual(len(loaded.spots), 3)
            self.assertEqual(len(loaded.sets), 1)
            self.assertTrue(loaded.sets[0].confirmed)
            self.assertEqual(loaded.sets[0].label_offsets["g1"], (30., -10.))
            self.assertEqual(loaded_style.font_size, 32)
            self.assertTrue(loaded_style.font_bold)
            self.assertEqual(loaded_style.line_width, 5)
            self.assertEqual(loaded_style.cover_rect, [10, 400, 240, 490])
            self.assertEqual(loaded_style.simulation_size, 9)
            self.assertEqual(loaded.image.calibration.reference_points, [[20, 460], [220, 460]])
            self.assertEqual(loaded.image.calibration.reference_value, 5)
            self.assertEqual(loaded.sets[0].label_offsets["zone"], (-40., -20.))
            np.testing.assert_allclose(loaded.sets[0].lattice.g1, group.lattice.g1)
            self.assertEqual(loaded.sets[0].indices()[2], group.indices()[2])

    def test_disabled_phases_do_not_appear_in_candidates(self):
        from test_core import cubic_session
        session, s = cubic_session()
        session.scale_tol = .2
        session.index_set(s)
        self.assertGreater(len(s.candidates), 0)
        session.phases[0].enabled = False
        session.index_set(s)
        self.assertEqual(s.candidates, [])

    def test_missing_predicted_spot_is_not_exported_as_measured(self):
        from test_core import cubic_session
        session, s = cubic_session()
        del s.lattice.members[2]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            session.write_csv(path)
            import csv
            with path.open(encoding="utf-8-sig") as handle:
                row = next(r for r in csv.reader(handle) if r and r[0] == "g1+g2")
            self.assertEqual(row[2], "")
            self.assertEqual(row[4], "")
