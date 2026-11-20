"""Bounded native interaction and publication-export verification."""
from pathlib import Path
import time
import traceback

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from saedindex.project import load_project, save_project
from saedindex.rendering import annotation_labels, export_figure
from saedindex.window import MainWindow

APP = QApplication([])
window = MainWindow()
window.show()
source = Path(r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000\1-0013.jpg")
out = Path(__file__).resolve().parent.parent / "verification"
deadline = time.monotonic() + 60
result = [1]
window.open_image(source)


def click_spot(index):
    spot = window.session.spots[index]
    QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton,
                     pos=window.canvas.to_widget(spot.x, spot.y).toPoint())
    APP.processEvents()


def verify():
    session = window.session
    assert session.image is not None
    original_scale = session.px_per_unit
    assert "5" in window.cal_evidence.text()
    assert "px" in window.cal_evidence.text()
    assert window.cal_thumbnail.pixmap() is not None
    primary = session.sets[0]
    pairs = list(primary.lattice.members)
    i = min(pairs, key=lambda k: np.linalg.norm(session.spots[k].xy - session.center))
    a = session.spots[i].xy - session.center
    j = max(pairs, key=lambda k: abs(np.linalg.det(
        np.column_stack([a, session.spots[k].xy - session.center]))))
    count = len(session.sets)
    window.add_group_button.click()
    click_spot(i)
    click_spot(j)
    assert len(session.sets) == count + 1
    window.activate_mode("delete_group")
    click_spot(i)
    assert len(session.sets) == count
    assert session.sets[0] is primary
    assert session.px_per_unit == original_scale

    window.current_set = 0
    window.refresh()
    window.stage_scale([[30, 980], [230, 980]])
    assert session.px_per_unit == original_scale
    assert not window._export_guard()
    window.reference_value.setValue(5)
    window.reference_pixels.setValue(original_scale * 5)
    bar_value = session.bar_value
    window.apply_reference()
    assert abs(session.px_per_unit - original_scale) < .002
    assert session.bar_value == bar_value
    assert not window._reference_dirty

    window.font_size.setValue(34)
    window.font_bold.setChecked(True)
    window.line_width.setValue(3.5)
    window.activate_mode("pan")
    for key in ("g1", "zone"):
        label = next(l for l in annotation_labels(session, window.style)
                     if l.set_index == 0 and l.key == key)
        start = window.canvas.to_widget(label.rect.center().x(), label.rect.center().y())
        target = start + QPointF(-15, -15)
        QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=start.toPoint())
        QTest.mouseMove(window.canvas, target.toPoint(), delay=20)
        QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=target.toPoint())
        assert key in primary.label_offsets
    window.simulation_box.setChecked(True)
    APP.processEvents()
    window.resize(1480, 960)
    APP.processEvents()
    assert window.sidebar.horizontalScrollBar().maximum() == 0
    assert window.right_panel.horizontalScrollBar().maximum() == 0
    window.grab().save(str(out / "revision-calibration-simulation.png"))
    window.figure_tabs.setCurrentIndex(1)
    APP.processEvents()
    window.grab().save(str(out / "revision-annotation-controls.png"))
    window.resize(1100, 800)
    APP.processEvents()
    assert window.canvas.width() >= 360
    assert window.sidebar.horizontalScrollBar().maximum() == 0
    assert window.right_panel.horizontalScrollBar().maximum() == 0
    window.grab().save(str(out / "revision-compact.png"))
    window.resize(1480, 960)
    window.figure_tabs.setCurrentIndex(2)
    window.preview_box.setChecked(True)
    APP.processEvents()
    window.grab().save(str(out / "revision-publication-preview.png"))
    export_figure(session, window.style, out / "revision-publication-draft.png")
    save_project(out / "revision-example.saed.json", session, window.style)
    restored, style = load_project(out / "revision-example.saed.json")
    assert style.font_bold and style.line_width == 3.5
    assert restored.sets[0].label_offsets == primary.label_offsets
    assert restored.image.calibration.reference_points == session.image.calibration.reference_points
    print(f"REVISION NATIVE OK: groups add/delete={count}, scale={session.px_per_unit:.5f}, "
          f"labels={list(primary.label_offsets)}, clean export and project restored.")
    result[0] = 0


def check():
    if window.worker is not None and time.monotonic() < deadline:
        QTimer.singleShot(100, check)
        return
    try:
        assert window.worker is None, "Analysis timed out"
        verify()
    except Exception:
        traceback.print_exc()
    finally:
        if window.worker:
            window.worker.wait()
        window.close()
        APP.quit()


QTimer.singleShot(100, check)
APP.exec()
raise SystemExit(result[0])
