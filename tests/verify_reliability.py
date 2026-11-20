"""Native diagnostics, compact layout, undo, clean export and saved provenance."""
from pathlib import Path
import tempfile

from PySide6.QtWidgets import QApplication

from saedindex.project import load_project, save_project
from saedindex.rendering import export_figure
from saedindex.window import MainWindow

APP = QApplication([])
window = MainWindow()
out = Path(__file__).resolve().parent.parent / "verification"
source = Path(r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000\1-0013.jpg")

try:
    window.show()
    window.open_image(source, background=False)
    session = window.session
    assert session.sets and session.sets[0].candidate.phase.name.startswith("达尔塔")
    window.tabs.setCurrentIndex(3)
    window.residual_box.setChecked(True)
    APP.processEvents()
    assert window.diagnostics_table.rowCount() == len(session.spots)
    assert window.canvas.simulation_diagnostics
    window.diagnostics_table.selectRow(1)
    assert window.canvas.selected == 1
    for width, height in ((1480, 960), (1100, 800)):
        window.resize(width, height)
        APP.processEvents()
        assert window.canvas.width() >= 360
        assert window.sidebar.horizontalScrollBar().maximum() == 0
        assert window.right_panel.horizontalScrollBar().maximum() == 0
        window.grab().save(str(out / f"reliability-diagnostics-{width}.png"))
    window.resize(1480, 960)
    window.markers_action.setChecked(False)
    APP.processEvents()
    assert not window.canvas.show_markers
    window.grab().save(str(out / "reliability-markers-hidden.png"))
    window.markers_action.setChecked(True)
    before = session.px_per_unit
    window.cal_value.setValue(before * 1.01)
    window.apply_calibration()
    window.undo()
    assert abs(session.px_per_unit - before) < 1e-8
    window.redo()
    assert abs(session.px_per_unit - before * 1.01) < 1e-4
    window.undo()
    window.preview_box.setChecked(True)
    APP.processEvents()
    window.grab().save(str(out / "reliability-publication-preview.png"))
    export_figure(session, window.style, out / "reliability-publication-draft.png")
    session.write_csv(out / "reliability-measurements.csv")
    path = out / "reliability-example.saed.json"
    save_project(path, session, window.style)
    restored, style = load_project(path)
    assert style.show_residuals
    assert restored.grouping_report == session.grouping_report
    assert restored.image.calibration.metadata_px_per_unit == session.image.calibration.metadata_px_per_unit
    with tempfile.TemporaryDirectory() as temp:
        compare = Path(temp) / "clean.png"
        style.show_residuals = False
        export_figure(restored, style, compare)
        assert compare.read_bytes() == (out / "reliability-publication-draft.png").read_bytes()
    window.preview_box.setChecked(False)
    window._sync_controls()
    window.refresh()
    APP.processEvents()
    window.grab().save(str(out / "reliability-diagnostics-final.png"))
    print("RELIABILITY NATIVE OK: diagnostics, residuals, marker toggle, undo/redo, "
          "two window sizes, project roundtrip and identical clean export.")
finally:
    window.close()
    APP.processEvents()
