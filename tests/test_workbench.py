import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from test_core import cubic_session

APP = QApplication.instance() or QApplication([])


class WorkbenchTests(unittest.TestCase):
    def window(self):
        from saedindex.window import MainWindow
        w = MainWindow(load_defaults=False)
        w.resize(1440, 900)
        w.show()
        APP.processEvents()
        self.addCleanup(w.close)
        return w

    def test_empty_window_and_export_disabled(self):
        w = self.window()
        self.assertIsNone(w.session.image)
        self.assertFalse(w.export_action.isEnabled())
        self.assertIsNotNone(w.canvas)

    def test_compact_window_keeps_image_canvas_usable(self):
        w = self.window()
        w.resize(1100, 800)
        APP.processEvents()
        self.assertGreaterEqual(w.canvas.width(), 360)
        self.assertEqual(w.sidebar.horizontalScrollBar().maximum(), 0)
        self.assertEqual(w.right_panel.horizontalScrollBar().maximum(), 0)

    def test_endpoint_coordinates_fit_in_compact_sidebar(self):
        w = self.window()
        w.resize(1100, 800)
        for control in w.reference_coordinates:
            control.setValue(1023.)
        APP.processEvents()
        for control in w.reference_coordinates:
            self.assertLessEqual(control.fontMetrics().horizontalAdvance(control.text()),
                                 control.lineEdit().contentsRect().width() - 4)

    def test_image_only_auto_calibration_and_spots(self):
        w = self.window()
        with tempfile.TemporaryDirectory() as temp:
            image = Image.new("L", (1024, 1024), 3)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("C:/Windows/Fonts/timesbd.ttf", 60)
            draw.text((30, 900), "5 1/nm", fill=255, font=font)
            draw.rectangle((30, 980, 220, 992), fill=255)
            for x, y in ((512, 450), (412, 450), (612, 450), (512, 350),
                         (512, 550), (612, 350), (412, 550)):
                draw.ellipse((x-4, y-4, x+4, y+4), fill=230)
            path = Path(temp) / "saed.png"
            image.save(path)
            w.open_image(path, background=False)
            self.assertAlmostEqual(w.session.px_per_unit, 38., delta=.5)
            self.assertIn("标尺", w.cal_source.text())
            self.assertGreaterEqual(len(w.session.spots), 7)
            self.assertTrue(w.export_action.isEnabled())

    def test_uncalibrated_image_does_not_keep_previous_scale_field(self):
        w = self.window()
        w.cal_value.setValue(42.)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unknown.png"
            Image.new("L", (256, 256)).save(path)
            w.open_image(path, background=False)
            self.assertIsNone(w.session.px_per_unit)
            self.assertEqual(w.cal_value.value(), 0.)

    def test_candidates_and_confirmation_invalidates_on_calibration_change(self):
        w = self.window()
        session, s = cubic_session()
        s.candidates = [s.custom]
        s.custom = None
        session.image.calibration.confident = True
        w.session = session
        w.refresh()
        self.assertEqual(w.candidate_table.rowCount(), 1)
        self.assertEqual(w.vector_table.rowCount(), 4)
        w.confirm_set()
        self.assertTrue(s.confirmed)
        w.cal_value.setValue(42.)
        w.apply_calibration()
        self.assertFalse(s.confirmed)

    def test_brightness_preserves_measurement_and_spot_move(self):
        w = self.window()
        session, s = cubic_session()
        w.session = session
        w.refresh()
        values = session.vector_rows(s)
        w.brightness.setValue(25)
        self.assertEqual(session.vector_rows(s), values)
        w.canvas.set_mode("move")
        p = w.canvas.to_widget(366, 256).toPoint()
        q = w.canvas.to_widget(370, 260).toPoint()
        QTest.mousePress(w.canvas, Qt.MouseButton.LeftButton, pos=p)
        QTest.mouseMove(w.canvas, q, delay=10)
        QTest.mouseRelease(w.canvas, Qt.MouseButton.LeftButton, pos=q)
        self.assertAlmostEqual(session.spots[0].x, 370, delta=2)
        self.assertAlmostEqual(session.spots[0].y, 260, delta=2)

    def test_background_worker_returns_error_without_mutating_original_session(self):
        from saedindex.window import AnalysisWorker
        from saedindex.session import Session
        session = Session()
        worker = AnalysisWorker(session, "Z:/not-a-real-folder/missing.png")
        spy = QSignalSpy(worker.failed)
        worker.start()
        for _ in range(100):
            APP.processEvents()
            if spy.count():
                break
            QTest.qWait(20)
        self.assertTrue(worker.wait(5000))
        self.assertEqual(spy.count(), 1)
        self.assertIsNone(session.image)


if __name__ == "__main__":
    unittest.main()
