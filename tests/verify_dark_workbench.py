"""Native dark workbench, async redetection, history and export checks."""
from pathlib import Path
import time

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from saedindex.project import load_project, save_project
from saedindex.rendering import annotation_labels, export_figure
from saedindex.window import MainWindow

APP = QApplication([])
window = MainWindow()
out = Path(__file__).resolve().parent.parent / 'verification'
source = Path(r'F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000\1-0013.jpg')


def wait_for_worker():
    deadline = time.monotonic() + 60
    while window.worker is not None and time.monotonic() < deadline:
        APP.processEvents()
        QTest.qWait(20)
    assert window.worker is None, 'Analysis timed out'


try:
    window.show()
    window.open_image(source)
    wait_for_worker()
    before = len(window.session.spots)
    scale = window.session.px_per_unit
    window.session.sets[0].confirmed = True
    window.sidebar_tabs.setCurrentIndex(1)
    window.min_diameter.setValue(4.)
    window.redetect_button.click()
    assert window.worker is not None
    wait_for_worker()
    after = len(window.session.spots)
    assert after < before and after > 10, (before, after)
    assert not any(s.confirmed for s in window.session.sets)
    assert window.session.px_per_unit == scale
    assert window.undo_action.isEnabled()
    window.undo()
    assert len(window.session.spots) == before
    assert window.session.sets[0].confirmed
    assert window.min_diameter.value() == 0
    window.redo()
    assert len(window.session.spots) == after
    assert window.min_diameter.value() == 4
    window.undo()
    window.session.sets[0].confirmed = False
    window.refresh()
    window.figure_tabs.setCurrentIndex(1)
    window.title_edit.setText('δ-NbNi3')
    window.title_changed()
    assert any('NbNi₃' in label.suffix for label in annotation_labels(window.session, window.style))
    window.simulation_box.setChecked(True)
    for width, height in ((1480, 960), (1100, 800)):
        window.resize(width, height)
        for tab in range(3):
            window.sidebar_tabs.setCurrentIndex(tab)
            APP.processEvents()
            assert window.canvas.width() >= 360, window.canvas.width()
            assert window.sidebar.horizontalScrollBar().maximum() == 0
            assert window.right_panel.horizontalScrollBar().maximum() == 0
            assert window.setting_scrolls[tab].horizontalScrollBar().maximum() == 0, (width, tab)
            window.grab().save(str(out / f'dark-workbench-{width}-tab{tab}.png'))
    window.resize(1480, 960)
    window.sidebar_tabs.setCurrentIndex(1)
    APP.processEvents()
    QTest.mouseClick(window.spots_button, Qt.MouseButton.LeftButton)
    assert not window.style.show_spots and window.canvas.show_markers
    assert window.canvas.simulations
    assert window.style.show_annotations
    APP.processEvents()
    window.grab().save(str(out / 'dark-workbench-rings-hidden.png'))
    export_figure(window.session, window.style, out / 'dark-workbench-figure.png')
    window.show_spots.setChecked(True)
    export_figure(window.session, window.style, out / 'dark-workbench-figure-check.png')
    assert (out / 'dark-workbench-figure.png').read_bytes() == (out / 'dark-workbench-figure-check.png').read_bytes()
    window.show_spots.setChecked(False)
    save_project(out / 'dark-workbench.saed.json', window.session, window.style)
    session, style = load_project(out / 'dark-workbench.saed.json')
    assert session.min_diameter == 0 and not style.show_spots
    assert style.chemical_subscripts and session.sets[0].title == 'δ-NbNi3'
    print(f'DARK WORKBENCH OK: {before} -> {after} spots, independent rings, formula subscripts, '
          'async redetection undo/redo, both layouts, project roundtrip and clean export.')
finally:
    if window.worker:
        window.worker.wait()
        APP.processEvents()
    window.close()
    APP.processEvents()
