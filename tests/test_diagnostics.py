import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication, QMessageBox

from test_core import cubic_session
from saedindex.calibration import Calibration, load_image
from saedindex.indexing import pair_candidates
from saedindex.rendering import FigureStyle, export_figure, paint_overlay, qimage_from_array

APP = QApplication.instance() or QApplication([])


class CalibrationConflictTests(unittest.TestCase):
    def image(self, temp):
        path = Path(temp) / "reference.png"
        image = Image.new("L", (512, 512), 4)
        draw = ImageDraw.Draw(image)
        draw.text((25, 400), "5 1/nm", fill=255,
                  font=ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 44))
        draw.rectangle((25, 480, 215, 489), fill=255)
        image.save(path)
        return path

    def test_conflicting_metadata_and_bar_are_held_for_review(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.image(temp)
            # Only the unavailable instrument metadata source is substituted;
            # image loading, bar detection and OCR are real.
            with patch("saedindex.calibration.find_companion", return_value=path.with_suffix(".dm3")), \
                 patch("saedindex.calibration._dm_calibration", return_value=(
                     Calibration(55., source="DM metadata", confident=True), {"dm_shape": [512, 512]})):
                cal = load_image(path).calibration
            self.assertIsNone(cal.px_per_unit)
            self.assertFalse(cal.confident)
            self.assertTrue(cal.conflict)
            self.assertEqual(cal.metadata_px_per_unit, 55.)
            self.assertAlmostEqual(cal.label_px_per_unit, 38., delta=.5)

    def test_consistent_metadata_retains_both_evidence_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.image(temp)
            with patch("saedindex.calibration.find_companion", return_value=path.with_suffix(".dm3")), \
                 patch("saedindex.calibration._dm_calibration", return_value=(
                     Calibration(38., source="DM metadata", confident=True), {"dm_shape": [512, 512]})):
                cal = load_image(path).calibration
            self.assertTrue(hasattr(cal, "metadata_px_per_unit"))
            self.assertEqual(cal.px_per_unit, 38.)
            self.assertTrue(cal.confident)
            self.assertFalse(cal.conflict)

    def test_manual_scale_resolves_conflict_and_undo_restores_it(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        window.session, _ = cubic_session()
        cal = window.session.image.calibration
        cal.conflict, cal.confident, cal.px_per_unit = True, False, None
        window.session.px_per_unit = None
        window.refresh()
        window.cal_value.setValue(40.)
        window.apply_calibration()
        self.assertFalse(cal.conflict)
        window.undo()
        self.assertTrue(cal.conflict)
        self.assertIsNone(window.session.px_per_unit)
        window.redo()
        self.assertFalse(cal.conflict)


class DiagnosticsTests(unittest.TestCase):
    def diagnostics(self, session, group):
        from importlib.util import find_spec
        self.assertIsNotNone(find_spec("saedindex.diagnostics"))
        from saedindex.diagnostics import group_diagnostics
        return group_diagnostics(session, group)

    def test_cost_components_sum_to_actual_ranking_cost(self):
        session, group = cubic_session()
        self.assertTrue(hasattr(group.candidate, "cost_parts"))
        self.assertAlmostEqual(sum(group.candidate.cost_parts.values()), group.candidate.cost)
        pairs = pair_candidates(session.phases, 2.5, 2.5, 90., 1)
        self.assertTrue(pairs)
        for candidate in pairs:
            self.assertAlmostEqual(sum(candidate.cost_parts.values()), candidate.cost)

    def test_diagnostics_recompute_residuals_and_keep_missing_closure_missing(self):
        session, group = cubic_session()
        session.spots[2].x += 2
        data = self.diagnostics(session, group)
        self.assertEqual(data["support"], 6)
        self.assertAlmostEqual(data["closure_px"], 2.)
        row = next(r for r in data["spots"] if r["spot"] == 2)
        self.assertAlmostEqual(row["residual_px"], 2.)
        del group.lattice.members[2]
        data = self.diagnostics(session, group)
        self.assertIsNone(data["closure_px"])
        self.assertEqual(data["support"], 5)

    def test_shared_nodes_are_not_reported_as_new_independent_support(self):
        session, group = cubic_session()
        second = session.set_from_spots(0, 1)
        data = self.diagnostics(session, second)
        self.assertEqual(data["new_support"], 0)
        self.assertEqual(data["shared_support"], 6)
        self.assertTrue(any("重复" in warning for warning in data["warnings"]))

    def test_uncalibrated_group_still_has_geometric_diagnostics(self):
        session, group = cubic_session()
        session.px_per_unit = None
        group.custom = None
        data = self.diagnostics(session, group)
        self.assertEqual(data["support"], 6)
        self.assertIsNone(data["scale_suggestion"])
        self.assertIsNotNone(data["closure_px"])
        self.assertAlmostEqual(data["closure_px"], 0.)

    def test_missing_reference_spot_does_not_export_predicted_angle_as_measured(self):
        import csv
        session, group = cubic_session()
        del group.lattice.members[0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "angles.csv"
            session.write_csv(path)
            with path.open(encoding="utf-8-sig") as handle:
                row = next(r for r in csv.reader(handle) if r and r[0] == "g2")
        self.assertNotEqual(row[2], "")
        self.assertEqual(row[5], "")

    def test_cif_scale_suggestion_has_correct_direction_and_does_not_apply(self):
        session, group = cubic_session()
        data = self.diagnostics(session, group)
        self.assertAlmostEqual(data["scale_suggestion"], 44.)
        self.assertEqual(session.px_per_unit, 40.)

    def test_cif_residuals_use_assigned_hkl_and_fixed_scientific_scale(self):
        session, group = cubic_session()
        self.diagnostics(session, group)
        from saedindex.diagnostics import simulation_matches
        reflections = session.simulation(group)[0]
        result = simulation_matches(session, group, reflections)
        row = next(r for r in result if r["spot"] == 0)
        self.assertAlmostEqual(row["residual_px"], 10.)
        self.assertFalse(row["matched"])
        self.assertEqual(session.px_per_unit, 40.)

    def test_csv_includes_diagnostics_and_calibration_provenance(self):
        session, group = cubic_session()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            session.write_csv(path)
            text = path.read_text(encoding="utf-8-sig")
        self.assertIn("诊断汇总", text)
        self.assertIn("评分分项", text)
        self.assertIn("晶格残差", text)


class DiagnosticsUiTests(unittest.TestCase):
    def window(self):
        from saedindex.window import MainWindow
        w = MainWindow(load_defaults=False)
        w.session, group = cubic_session()
        w.session.image.calibration.confident = True
        w.refresh()
        self.addCleanup(w.close)
        return w, group

    def test_diagnostic_table_selects_real_spot_and_tracks_edits(self):
        w, group = self.window()
        self.assertTrue(hasattr(w, "diagnostics_table"))
        self.assertEqual(w.diagnostics_table.rowCount(), len(w.session.spots))
        w.diagnostics_table.selectRow(2)
        self.assertEqual(w.canvas.selected, int(w.diagnostics_table.item(2, 0).text()))
        w._begin_edit()
        w.session.spots[2].x += 2
        w.geometry_changed()
        self.assertIn("2.00", w.diagnostics_summary.text())
        w.undo()
        self.assertIn("0.00", w.diagnostics_summary.text())

    def test_scale_suggestion_requires_explicit_application_and_is_undoable(self):
        w, group = self.window()
        self.assertTrue(hasattr(w, "apply_cif_scale"))
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            w.apply_cif_scale()
        self.assertEqual(w.session.px_per_unit, 40.)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            w.apply_cif_scale()
        self.assertAlmostEqual(w.session.px_per_unit, 44.)
        self.assertIn("CIF", w.session.image.calibration.source)
        self.assertFalse(w.session.sets[0].confirmed)
        w.undo()
        self.assertAlmostEqual(w.session.px_per_unit, 40.)

    def test_diagnostics_markers_never_leak_to_publication_or_hidden_markers(self):
        w, group = self.window()
        self.assertTrue(hasattr(w, "residual_box"))
        group.show_sim = True
        w.refresh()
        w.residual_box.setChecked(True)
        w.canvas.refresh_simulations()
        self.assertTrue(w.style.show_residuals)
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.png", Path(temp) / "b.png"
            export_figure(w.session, w.style, first)
            w.style.show_residuals = False
            export_figure(w.session, w.style, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
        def render(markers):
            image = qimage_from_array(np.zeros((512, 512), np.uint8))
            painter = QPainter(image)
            paint_overlay(painter, w.session, w.style, editing=True,
                          simulations=w.canvas.simulations, markers=markers)
            painter.end()
            return bytes(image.constBits())
        w.style.show_residuals = True
        shown = render(True)
        hidden = render(False)
        w.style.show_residuals = False
        self.assertEqual(hidden, render(False))
        self.assertNotEqual(shown, hidden)

    def test_residual_controls_restore_with_style(self):
        w, _ = self.window()
        w.style.show_residuals = True
        w.style.free_scale = True
        w._sync_controls()
        self.assertTrue(w.residual_box.isChecked())
        self.assertFalse(w.free_scale_box.isEnabled())

    def test_delete_group_clears_old_analysis_report_and_undo_restores_it(self):
        w, _ = self.window()
        w.session.grouping_report = [{"accepted": True, "new_spots": 6, "reason": "自动分析"}]
        w.delete_set()
        self.assertEqual(w.session.grouping_report, [])
        w.undo()
        self.assertEqual(len(w.session.grouping_report), 1)

    def test_diagnostics_roundtrip_and_legacy_optional_fields(self):
        import json
        from saedindex.project import load_project, save_project
        w, _ = self.window()
        w.session.phases = []
        w.session.sets = []
        w.style.show_residuals = True
        w.session.grouping_report = [{"accepted": False, "new_spots": 4, "reason": "独立点不足"}]
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            Image.new("L", (512, 512), 0).save(image)
            w.session.image.path = image
            path = Path(temp) / "roundtrip.saed.json"
            save_project(path, w.session, w.style)
            restored, style = load_project(path)
            self.assertEqual(restored.grouping_report, w.session.grouping_report)
            self.assertTrue(style.show_residuals)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("grouping_report")
            payload["style"].pop("show_residuals")
            for key in ("metadata_px_per_unit", "label_px_per_unit", "label_confidence", "conflict"):
                payload["calibration"].pop(key)
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored, style = load_project(path)
            self.assertEqual(restored.grouping_report, [])
            self.assertFalse(style.show_residuals)


if __name__ == "__main__":
    unittest.main()
