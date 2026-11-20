import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from test_core import cubic_session
from saedindex.canvas import PatternCanvas
from saedindex.rendering import (
    FigureStyle, annotation_labels, export_figure, paint_overlay, qimage_from_array,
)

APP = QApplication.instance() or QApplication([])


class EditingTests(unittest.TestCase):
    def canvas(self):
        session, group = cubic_session()
        canvas = PatternCanvas()
        canvas.resize(700, 700)
        canvas.bind(session, FigureStyle(replace_bar=False))
        canvas.show()
        APP.processEvents()
        self.addCleanup(canvas.close)
        return canvas, session, group

    def click(self, canvas, xy):
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_widget(*xy).toPoint())

    def test_deleting_second_group_uses_identity_not_numpy_equality(self):
        session, first = cubic_session()
        second = session.set_from_spots(0, 1)
        session.remove_set(second)
        self.assertEqual(len(session.sets), 1)
        self.assertIs(session.sets[0], first)

    def test_build_group_without_center_is_a_clear_validation_error(self):
        session, _ = cubic_session()
        session.center = None
        with self.assertRaisesRegex(ValueError, "中心"):
            session.set_from_spots(0, 1)

    def test_mouse_delete_group_keeps_spots_and_other_groups(self):
        canvas, session, group = self.canvas()
        session.set_from_spots(0, 1)
        canvas.active_set = 1
        self.assertTrue(hasattr(canvas, "group_delete_requested"))
        spy = QSignalSpy(canvas.group_delete_requested)
        canvas.set_mode("delete_group")
        self.click(canvas, (366, 256))
        self.assertEqual(spy.count(), 1)
        self.assertEqual(spy.at(0)[0], 1)
        self.assertEqual(len(session.spots), 6)

    def test_scale_mouse_selection_retains_both_endpoints(self):
        canvas, _, _ = self.canvas()
        self.assertTrue(hasattr(canvas, "reference_selected"))
        spy = QSignalSpy(canvas.reference_selected)
        canvas.set_mode("scale")
        self.click(canvas, (30, 460))
        self.click(canvas, (230, 460))
        self.assertEqual(spy.count(), 1)
        np.testing.assert_allclose(spy.at(0)[0], [[30, 460], [230, 460]], atol=1)

    def test_label_drag_in_default_pan_mode_applies_release_position(self):
        canvas, session, group = self.canvas()
        label = next(l for l in annotation_labels(session, canvas.style) if l.key == "zone")
        start = label.rect.center()
        end = start + QPointF(-25, -30)
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_widget(start.x(), start.y()).toPoint())
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton,
                           pos=canvas.to_widget(end.x(), end.y()).toPoint())
        self.assertIn("zone", group.label_offsets)
        moved = next(l for l in annotation_labels(session, canvas.style) if l.key == "zone")
        np.testing.assert_allclose(
            [moved.rect.x() - label.rect.x(), moved.rect.y() - label.rect.y()],
            [-25, -30], atol=1.5)

    def test_tools_click_inspect_center_add_delete_and_pair(self):
        canvas, session, _ = self.canvas()
        inspect = QSignalSpy(canvas.inspected)
        canvas.set_mode("inspect")
        self.click(canvas, (366, 256))
        self.assertEqual(inspect.at(0)[0], 0)
        canvas.set_mode("add")
        self.click(canvas, (80, 90))
        self.assertEqual(len(session.spots), 7)
        canvas.set_mode("delete")
        self.click(canvas, (80, 90))
        self.assertEqual(len(session.spots), 6)
        canvas.set_mode("center")
        self.click(canvas, (250, 250))
        np.testing.assert_allclose(session.center, [250, 250], atol=1)
        pairs = QSignalSpy(canvas.pair_selected)
        canvas.set_mode("pair")
        self.click(canvas, (366, 256))
        self.click(canvas, (256, 146))
        self.assertEqual(pairs.count(), 1)
        self.assertEqual(list(pairs.at(0)), [0, 1])

    def test_draft_scale_change_does_not_apply_until_requested(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        window.session, _ = cubic_session()
        window.refresh()
        self.assertTrue(hasattr(window, "stage_scale"))
        window.stage_scale([[30, 460], [230, 460]])
        window.reference_value.setValue(5)
        self.assertEqual(window.session.px_per_unit, 40)
        window.reference_pixels.setValue(210)
        self.assertEqual(window.session.px_per_unit, 40)
        window.apply_reference()
        self.assertAlmostEqual(window.session.px_per_unit, 42)
        self.assertAlmostEqual(window.session.image.calibration.reference_value, 5)

    def test_calibration_endpoints_can_be_dragged_without_applying_scale(self):
        canvas, session, _ = self.canvas()
        self.assertTrue(hasattr(canvas, "reference_selected"))
        canvas.calibration_points = [[30, 460], [230, 460]]
        spy = QSignalSpy(canvas.reference_selected)
        canvas.set_mode("scale")
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_widget(230, 460).toPoint())
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton,
                           pos=canvas.to_widget(250, 455).toPoint())
        self.assertEqual(spy.count(), 1)
        np.testing.assert_allclose(spy.at(0)[0][1], [250, 455], atol=1)
        self.assertEqual(session.px_per_unit, 40)

    def test_scientific_preview_keeps_original_scale_pixels(self):
        canvas, session, _ = self.canvas()
        session.image.gray[450:480, 20:240] = 255
        session.image.calibration.label_rect = [20, 450, 240, 480]
        session.bar_value = 5
        canvas.style.replace_bar = True
        canvas.refresh_background()
        self.assertEqual(canvas.background.pixelColor(40, 460).red(), 255)

    def test_cover_mask_does_not_change_scientific_exclusion(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        window.session, _ = cubic_session()
        window.session.image.calibration.label_rect = [20, 450, 240, 480]
        window.refresh()
        window.set_mask([10, 400, 260, 490])
        self.assertEqual(window.session.exclude, [[20, 450, 240, 480]])
        self.assertEqual(window.style.cover_rect, [10, 400, 260, 490])
        self.assertEqual(window.session.px_per_unit, 40)

    def test_line_width_and_bold_controls_update_style(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        self.assertTrue(hasattr(window, "line_width"))
        self.assertTrue(hasattr(window, "font_bold"))
        window.line_width.setValue(5.5)
        window.font_bold.setChecked(True)
        self.assertEqual(window.style.line_width, 5.5)
        self.assertTrue(window.style.font_bold)

    def test_unindexed_groups_do_not_appear_in_publication_export(self):
        session, group = cubic_session()
        group.custom = None
        image = qimage_from_array(session.image.gray)
        painter = QPainter(image)
        paint_overlay(painter, session, FigureStyle(replace_bar=False), editing=False)
        painter.end()
        pixels = np.array(image.constBits())
        self.assertEqual(pixels.max(), 0)

    def test_export_is_independent_of_editing_markers(self):
        session, group = cubic_session()
        session.bar_value = 2.5
        group.show_sim = True
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a.png", Path(temp) / "b.png"
            style = FigureStyle()
            export_figure(session, style, a)
            style.show_spots = False
            style.show_center = False
            style.show_simulation = False
            export_figure(session, style, b)
            with Image.open(a) as ia, Image.open(b) as ib:
                np.testing.assert_array_equal(np.asarray(ia), np.asarray(ib))

    def test_simulation_marks_remain_visible_at_fit_zoom(self):
        canvas, session, group = self.canvas()
        self.assertTrue(hasattr(canvas.style, "simulation_size"))
        group.show_sim = True
        style = FigureStyle(show_spots=False, show_center=False, show_annotations=False,
                            replace_bar=False)
        image = qimage_from_array(np.zeros((256, 256), np.uint8)).convertToFormat(
            canvas.background.Format.Format_RGB888)
        painter = QPainter(image)
        painter.scale(.5, .5)
        paint_overlay(painter, session, style, editing=True,
                      simulations={0: [((1, 0, 0), 200, 200, 1., True)]})
        painter.end()
        rgb = np.asarray(image.constBits()).reshape(256, image.bytesPerLine())
        self.assertGreater(np.count_nonzero(rgb[90:111, 90*3:111*3]), 100)

    def test_enabling_simulation_leaves_publication_preview(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        window.session, _ = cubic_session()
        window.refresh()
        window.preview_box.setChecked(True)
        window.simulation_box.setChecked(True)
        self.assertFalse(window.style.publication_preview)
        self.assertTrue(window.selected_set.show_sim)
        self.assertTrue(window.canvas.simulations)

    def test_pending_scale_cannot_confirm_an_old_calibration(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        window.session, group = cubic_session()
        window.session.image.calibration.confident = True
        window.refresh()
        window.stage_scale([[20, 460], [220, 460]])
        window.confirm_set()
        self.assertFalse(group.confirmed)
        self.assertIn("未应用", window.statusBar().currentMessage())

    def test_mask_drag_and_pan_use_final_release_position(self):
        canvas, _, _ = self.canvas()
        masks = QSignalSpy(canvas.mask_selected)
        canvas.set_mode("mask")
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_widget(20, 420).toPoint())
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton,
                           pos=canvas.to_widget(220, 480).toPoint())
        self.assertEqual(masks.count(), 1)
        np.testing.assert_allclose(masks.at(0)[0], [20, 420, 220, 480], atol=1)
        canvas.set_mode("pan")
        QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=canvas.rect().center())
        QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton,
                           pos=canvas.rect().center() + QPointF(30, 20).toPoint())
        np.testing.assert_allclose([canvas.pan.x(), canvas.pan.y()], [30, 20], atol=1)

    def test_reference_units_validate_before_changing_measurements(self):
        session, group = cubic_session()
        self.assertTrue(hasattr(session, "apply_reference"))
        session.bar_value = 2.5
        session.apply_reference([[10, 460], [210, 460]], .5, "1/A")
        self.assertEqual(session.px_per_unit, 40)
        self.assertEqual(session.bar_value, 2.5)
        for points, value, unit in [
            ([[10, 460], [210, 460]], 5, "nm"),
            ([[10, 460], [10, 460]], 5, "1/nm"),
            ([[10, 460], [800, 460]], 5, "1/nm"),
            ([[10, 460], [210, 460]], float("nan"), "1/nm"),
        ]:
            with self.subTest(points=points, value=value, unit=unit):
                with self.assertRaises(ValueError):
                    session.apply_reference(points, value, unit)
                self.assertEqual(session.px_per_unit, 40)

    def test_restore_reference_discards_draft_without_changing_calibration(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        window.session, _ = cubic_session()
        window.session.apply_reference([[20, 460], [220, 460]], 5, "1/nm")
        window.refresh()
        window.stage_scale([[20, 460], [240, 460]])
        self.assertTrue(window._reference_dirty)
        self.assertTrue(hasattr(window, "restore_reference"))
        window.restore_reference()
        self.assertFalse(window._reference_dirty)
        self.assertEqual(window.reference_pixels.value(), 200)
        self.assertEqual(window.session.px_per_unit, 40)

    def window(self):
        from saedindex.window import MainWindow
        window = MainWindow(load_defaults=False)
        self.addCleanup(window.close)
        window.session, _ = cubic_session()
        window.refresh()
        return window

    def test_new_image_does_not_inherit_previous_cover_rect(self):
        window = self.window()
        window.set_mask([10, 400, 260, 490])
        session, _ = cubic_session()
        window._result(session)
        self.assertIsNone(window.style.cover_rect)

    def test_clicking_spot_in_move_mode_keeps_confirmation(self):
        canvas, session, group = self.canvas()
        group.confirmed = True
        changed = QSignalSpy(canvas.changed)
        canvas.set_mode("move")
        self.click(canvas, (366, 256))
        self.assertEqual(changed.count(), 0)
        self.assertTrue(group.confirmed)

    def test_deleted_group_can_be_restored_in_place(self):
        window = self.window()
        first_name = window.session.sets[0].name
        second = window.session.set_from_spots(0, 1)
        second.label_offsets["zone"] = (5., 6.)
        window.refresh()
        window.delete_group_at(0)
        self.assertIs(window.session.sets[0], second)
        self.assertTrue(window.undo_action.isEnabled())
        window.undo()
        self.assertEqual([s.name for s in window.session.sets], [first_name, second.name])
        self.assertEqual(window.session.sets[1].label_offsets["zone"], (5., 6.))
        self.assertIs(window.session.sets[0].candidate.phase, window.session.phases[0])
        self.assertEqual(window.current_set, 0)
        self.assertFalse(window.undo_action.isEnabled())
        self.assertTrue(window.redo_action.isEnabled())

    def shown_window(self):
        window = self.window()
        window.resize(1480, 960)
        window.show()
        APP.processEvents()
        return window

    def drag(self, canvas, start, end):
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=canvas.to_widget(*start).toPoint())
        QTest.mouseMove(canvas, canvas.to_widget(*end).toPoint(), delay=10)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=canvas.to_widget(*end).toPoint())

    def test_spot_drag_can_be_undone_and_redone(self):
        window = self.shown_window()
        window.session.sets[0].confirmed = True
        window.activate_mode("move")
        self.drag(window.canvas, (366, 256), (380, 270))
        spot = window.session.spots[0]
        self.assertGreater(spot.x, 370)
        self.assertFalse(window.session.sets[0].confirmed)
        window.undo()
        spot = window.session.spots[0]
        self.assertAlmostEqual(spot.x, 366, places=3)
        self.assertAlmostEqual(spot.y, 256, places=3)
        self.assertTrue(window.session.sets[0].confirmed)
        window.redo()
        self.assertGreater(window.session.spots[0].x, 370)
        self.assertFalse(window.redo_action.isEnabled())

    def test_click_without_moving_records_no_history(self):
        window = self.shown_window()
        window.activate_mode("move")
        self.click(window.canvas, (366, 256))
        self.assertFalse(window.undo_action.isEnabled())

    def test_applied_scale_can_be_undone(self):
        window = self.shown_window()
        before = window.session.px_per_unit
        window.stage_scale([[30, 460], [230, 460]])
        window.reference_value.setValue(2.)
        window.apply_reference()
        self.assertAlmostEqual(window.session.px_per_unit, 100.)
        window.undo()
        self.assertAlmostEqual(window.session.px_per_unit, before)
        cal = window.session.image.calibration
        self.assertAlmostEqual(cal.px_per_unit, before)
        self.assertIsNone(cal.reference_points)
        window.redo()
        self.assertAlmostEqual(window.session.px_per_unit, 100.)
        self.assertEqual(window.session.image.calibration.reference_points,
                         [[30., 460.], [230., 460.]])

    def test_undo_first_discards_pending_scale_draft(self):
        window = self.shown_window()
        window.delete_group_at(0)
        window.stage_scale([[30, 460], [230, 460]])
        self.assertTrue(window._reference_dirty)
        window.undo()
        self.assertFalse(window._reference_dirty)
        self.assertEqual(len(window.session.sets), 0)
        window.undo()
        self.assertEqual(len(window.session.sets), 1)

    def test_label_move_is_undoable_and_new_edit_clears_redo(self):
        window = self.shown_window()
        canvas = window.canvas
        label = next(l for l in annotation_labels(window.session, window.style) if l.key == "zone")
        start = label.rect.center()
        self.drag(canvas, (start.x(), start.y()), (start.x() + 30, start.y() + 20))
        self.assertIn("zone", window.session.sets[0].label_offsets)
        window.undo()
        self.assertNotIn("zone", window.session.sets[0].label_offsets)
        self.assertTrue(window.redo_action.isEnabled())
        window.activate_mode("move")
        self.drag(canvas, (366, 256), (380, 270))
        self.assertFalse(window.redo_action.isEnabled())

    def test_new_image_clears_history(self):
        window = self.window()
        window.delete_group_at(0)
        session, _ = cubic_session()
        window._result(session)
        self.assertFalse(window.undo_action.isEnabled())
        self.assertFalse(window.redo_action.isEnabled())

    def test_marker_toggle_hides_recognition_marks_but_keeps_annotations(self):
        window = self.shown_window()
        canvas = window.canvas
        ring = canvas.to_widget(366 + 5.5, 256).toPoint()
        canvas.selected = None
        shown = canvas.grab().toImage()
        window.markers_action.trigger()
        self.assertFalse(canvas.show_markers)
        APP.processEvents()
        hidden = canvas.grab().toImage()
        self.assertNotEqual(shown.pixelColor(ring), hidden.pixelColor(ring))
        self.assertTrue(annotation_labels(window.session, window.style))
        window.activate_mode("move")
        self.assertTrue(canvas.show_markers)
        self.assertTrue(window.markers_action.isChecked())

    def test_publication_preview_leaves_hidden_spot_tools(self):
        window = self.window()
        window.activate_mode("add")
        window.preview_box.setChecked(True)
        self.assertEqual(window.canvas.mode, "pan")
        self.assertTrue(window.mode_actions["pan"].isChecked())
        window.activate_mode("labels")
        window.preview_box.setChecked(True)
        self.assertEqual(window.canvas.mode, "labels")

    def test_endpoint_drag_updates_panel_live_and_small_mask_clears(self):
        canvas, _, _ = self.canvas()
        canvas.calibration_points = [[30, 460], [230, 460]]
        spy = QSignalSpy(canvas.reference_selected)
        canvas.set_mode("scale")
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton,
                         pos=canvas.to_widget(230, 460).toPoint())
        QTest.mouseMove(canvas, canvas.to_widget(240, 460).toPoint(), delay=10)
        self.assertGreaterEqual(spy.count(), 1)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton,
                           pos=canvas.to_widget(240, 460).toPoint())
        canvas.set_mode("mask")
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=canvas.to_widget(20, 20).toPoint())
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=canvas.to_widget(21, 21).toPoint())
        self.assertIsNone(canvas._mask)

    def test_auto_endpoints_follow_measured_bar_core(self):
        from saedindex.calibration import _bar_result
        bar = _bar_result(0, 10, 100, 190, 2, 2, 20)
        self.assertAlmostEqual(bar["start"] + bar["pixels"], 100 + 190 + .365, places=3)
        self.assertAlmostEqual(bar["start"], 100 - .365, places=3)


if __name__ == "__main__":
    unittest.main()
