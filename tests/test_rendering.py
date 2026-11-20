import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from test_core import cubic_session

APP = QApplication.instance() or QApplication([])


class RenderingTests(unittest.TestCase):
    def test_fonts_have_real_glyphs_on_offscreen_windows(self):
        from saedindex.rendering import ensure_fonts
        from PySide6.QtGui import QFontMetricsF
        ensure_fonts()
        self.assertTrue(QFontMetricsF(QFont("Times New Roman", 30)).inFont("1"))
        self.assertTrue(QFontMetricsF(QFont("Microsoft YaHei UI", 10)).inFont("晶"))

    def test_display_adjustments_do_not_modify_input(self):
        from saedindex.rendering import display_array, FigureStyle
        data = np.arange(256, dtype=np.uint8).reshape(16, 16)
        original = data.copy()
        result = display_array(data, FigureStyle(brightness=20, contrast=1.5, gamma=1.2))
        np.testing.assert_array_equal(data, original)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.array_equal(data, result))

    def test_bar_pixel_length_is_calibrated(self):
        from saedindex.rendering import FigureStyle, bar_geometry
        session, _ = cubic_session()
        session.bar_value = 2.5
        rect, _ = bar_geometry(session, FigureStyle())
        self.assertAlmostEqual(rect.width(), 100.)

    def test_overbar_glyph_keeps_sign_out_of_text(self):
        from saedindex.rendering import index_runs
        runs = index_runs((1, -2, 0), "(", QFont("Times New Roman", 30))
        self.assertEqual("".join(r[0] for r in runs), "(120)")
        self.assertEqual([r[1] for r in runs], [False, False, True, False, False])

    def test_export_preserves_resolution_and_writes_dpi(self):
        from saedindex.rendering import FigureStyle, export_figure
        session, _ = cubic_session()
        session.bar_value = 2.5
        original = session.image.gray.copy()
        with tempfile.TemporaryDirectory() as temp:
            for suffix in ("png", "tif"):
                path = Path(temp) / f"figure.{suffix}"
                export_figure(session, FigureStyle(), path)
                with Image.open(path) as image:
                    self.assertEqual(image.size, (512, 512))
                    self.assertAlmostEqual(image.info["dpi"][0], 600, delta=1)
                    self.assertGreater(np.asarray(image).max(), 0)
        np.testing.assert_array_equal(session.image.gray, original)

    def test_annotation_bounds_stay_inside_image(self):
        from saedindex.rendering import FigureStyle, annotation_labels
        session, s = cubic_session()
        s.label_offsets["g1"] = (1000., 1000.)
        for label in annotation_labels(session, FigureStyle(font_size=32)):
            self.assertGreaterEqual(label.rect.left(), 0)
            self.assertGreaterEqual(label.rect.top(), 0)
            self.assertLessEqual(label.rect.right(), 512)
            self.assertLessEqual(label.rect.bottom(), 512)


if __name__ == "__main__":
    unittest.main()
