import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication, QGroupBox

from test_core import cubic_session
from saedindex.pattern import detect_spots
from saedindex.window import MainWindow

APP = QApplication.instance() or QApplication([])


def bright_points():
    yy, xx = np.mgrid[:160, :160]
    raw = np.random.default_rng(35).normal(10, .15, xx.shape)
    for x, y, sigma in ((35, 40, .65), (80, 80, 2.6), (125, 115, 6.)):
        raw += 120 * np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma**2))
    return raw.astype(np.float32)


class DetectionControlsTests(unittest.TestCase):
    def test_size_filter_rejects_tiny_points_but_keeps_larger_peak(self):
        raw = bright_points()
        original = raw.copy()
        points = detect_spots(raw, min_diameter=4., max_diameter=10.)
        self.assertTrue(any(np.linalg.norm(p.xy - [80, 80]) < 1 for p in points))
        self.assertFalse(any(np.linalg.norm(p.xy - [35, 40]) < 3 for p in points))
        self.assertFalse(any(np.linalg.norm(p.xy - [125, 115]) < 3 for p in points))
        np.testing.assert_array_equal(raw, original)

    def test_disabled_size_filter_keeps_existing_behavior(self):
        old = detect_spots(bright_points())
        new = detect_spots(bright_points(), min_diameter=0, max_diameter=0)
        self.assertEqual([(s.x, s.y) for s in old], [(s.x, s.y) for s in new])

    def test_invalid_size_range_is_rejected(self):
        for args in ({'min_diameter': 8, 'max_diameter': 3},
                     {'min_diameter': -1}, {'smoothing': float('nan')}):
            with self.assertRaises(ValueError):
                detect_spots(bright_points(), **args)

    def test_session_detection_uses_size_settings(self):
        session, _ = cubic_session()
        session.image.raw = bright_points()
        session.min_diameter, session.max_diameter = 4., 10.
        session.detect()
        self.assertFalse(any(np.linalg.norm(p.xy - [35, 40]) < 3 for p in session.spots))
        self.assertTrue(any(np.linalg.norm(p.xy - [80, 80]) < 1 for p in session.spots))

    def test_project_retains_detection_settings_and_legacy_defaults(self):
        from saedindex.session import Session
        from saedindex.rendering import FigureStyle
        from saedindex.project import save_project, load_project
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.fromarray(np.zeros((64, 64), np.uint8)).save(root / 'image.png')
            session = Session()
            session.open(root / 'image.png')
            session.min_diameter, session.max_diameter, session.smoothing = 4., 16., 1.8
            path = root / 'test.saed.json'
            save_project(path, session, FigureStyle(show_spots=False, chemical_subscripts=False))
            restored, style = load_project(path)
            self.assertEqual((restored.min_diameter, restored.max_diameter, restored.smoothing), (4.,16.,1.8))
            self.assertFalse(style.show_spots)
            self.assertFalse(style.chemical_subscripts)
            data = json.loads(path.read_text(encoding='utf-8'))
            for key in ('min_diameter', 'max_diameter', 'smoothing'):
                del data['settings'][key]
            del data['style']['chemical_subscripts']
            path.write_text(json.dumps(data), encoding='utf-8')
            legacy, style = load_project(path)
            self.assertEqual((legacy.min_diameter, legacy.max_diameter, legacy.smoothing), (0.,0.,1.2))
            self.assertTrue(style.chemical_subscripts)


class WorkbenchControlsTests(unittest.TestCase):
    def window(self):
        window = MainWindow(load_defaults=False)
        window.session, _ = cubic_session()
        window.refresh()
        self.addCleanup(window.close)
        return window

    def test_spot_toggle_is_independent_and_synchronized(self):
        w = self.window()
        w.spots_action.setChecked(False)
        self.assertFalse(w.style.show_spots)
        self.assertFalse(w.show_spots.isChecked())
        self.assertTrue(w.canvas.show_markers)
        self.assertTrue(w.style.show_simulation)
        self.assertTrue(w.style.show_annotations)
        w.show_spots.setChecked(True)
        self.assertTrue(w.spots_action.isChecked())
        w.spots_action.setChecked(False)
        w.markers_action.setChecked(False)
        w.markers_action.setChecked(True)
        self.assertFalse(w.style.show_spots)

    def test_hiding_rings_preserves_simulation_pixels_and_clean_export(self):
        from PySide6.QtGui import QPainter
        from saedindex.rendering import paint_overlay, qimage_from_array, export_figure
        w = self.window()
        w.session.sets[0].show_sim = True
        simulations = {0: [((1,0,0),100.,100.,1.,True)]}
        def rendered():
            image = qimage_from_array(np.zeros((512,512),np.uint8))
            painter = QPainter(image)
            paint_overlay(painter,w.session,w.style,editing=True,simulations=simulations)
            painter.end()
            return image
        shown = rendered()
        w.spots_action.setChecked(False)
        hidden = rendered()
        self.assertNotEqual(shown.pixelColor(371,256),hidden.pixelColor(371,256))
        self.assertEqual(shown.pixelColor(107,100),hidden.pixelColor(107,100))
        self.assertGreater(hidden.pixelColor(107,100).lightness(),0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'figure.png'
            export_figure(w.session,w.style,path)
            clean = path.read_bytes()
            w.spots_action.setChecked(True)
            export_figure(w.session,w.style,path)
            self.assertEqual(clean,path.read_bytes())

    def test_changed_parameters_remain_pending_after_other_ui_refresh(self):
        w = self.window()
        w.min_diameter.setValue(5)
        w.refresh()
        self.assertIn('尚未重新识别',w.detection_status.text())

    def test_invalid_sizes_do_not_start_worker(self):
        w = self.window()
        w.min_diameter.setValue(10)
        w.max_diameter.setValue(3)
        w.analyse()
        self.assertIsNone(w.worker)

    def test_sidebar_groups_and_detection_controls(self):
        w = self.window()
        self.assertEqual([w.sidebar_tabs.tabText(i) for i in range(3)], ['定标','识别','显示'])
        self.assertGreaterEqual(len(w.findChildren(QGroupBox)), 7)
        w.min_diameter.setValue(5)
        w.max_diameter.setValue(20)
        w.smoothing.setValue(1.8)
        w._read_settings()
        self.assertEqual((w.session.min_diameter, w.session.max_diameter, w.session.smoothing), (5,20,1.8))

    def test_pending_calibration_blocks_redetection(self):
        w = self.window()
        w._reference_dirty = True
        w.analyse()
        self.assertIsNone(w.worker)

    def test_redetection_result_is_undoable_with_parameters(self):
        w = self.window()
        w.session.sets[0].confirmed = True
        original = len(w.session.spots)
        w._analysis_before = w._snapshot()
        w._analysis_new_image = False
        from saedindex.window import AnalysisWorker
        worker = AnalysisWorker(w.session)
        self.assertIs(worker.session.phases[0], w.session.phases[0])
        worker.session.spots = []
        worker.session.sets = []
        worker.session.min_diameter = 7
        w._result(worker.session)
        self.assertEqual(len(w.session.spots), 0)
        w.undo()
        self.assertEqual(len(w.session.spots), original)
        self.assertTrue(w.session.sets[0].confirmed)
        self.assertEqual(w.session.min_diameter, 0)
        self.assertEqual(w.min_diameter.value(), 0)
        w.redo()
        self.assertEqual(w.session.min_diameter, 7)
        self.assertEqual(w.min_diameter.value(), 7)

    def test_chemical_subscripts_do_not_rewrite_names_or_indices(self):
        from saedindex.rendering import chemical_text, annotation_labels
        self.assertEqual(chemical_text('NbNi3'), 'NbNi₃')
        self.assertEqual(chemical_text('δ-Al2O3'), 'δ-Al₂O₃')
        self.assertEqual(chemical_text('Fe2(SO4)3'), 'Fe₂(SO₄)₃')
        self.assertEqual(chemical_text('Sample 123 [001]'), 'Sample 123 [001]')
        w = self.window()
        group = w.session.sets[0]
        zone = next(l for l in annotation_labels(w.session,w.style) if l.key=='zone')
        self.assertEqual(zone.suffix.strip(), group.candidate.phase.name)
        self.assertTrue(zone.subscript)
        w.phase_subscript.setChecked(False)
        group.title = 'NbNi3'
        label = next(l for l in annotation_labels(w.session,w.style) if l.key=='zone')
        self.assertEqual(label.suffix, ' NbNi₃')
        self.assertFalse(label.subscript)
        self.assertEqual(group.title, 'NbNi3')
        w.chemical_subscripts.setChecked(False)
        label = next(l for l in annotation_labels(w.session,w.style) if l.key=='zone')
        self.assertEqual(label.suffix, ' NbNi3')

    def test_export_button_is_next_to_publication_preview(self):
        w = self.window()
        self.assertTrue(hasattr(w, 'preview_export_button'))
        self.assertEqual(w.preview_export_button.text(), '导出标定图')


if __name__ == '__main__':
    unittest.main()
