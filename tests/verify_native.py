"""Bounded native Qt smoke test for the background import and figure workflow."""
from pathlib import Path
import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from saedindex.window import MainWindow
from saedindex.project import save_project, load_project

app = QApplication([])
window = MainWindow()
window.show()
source = Path("C:/Users/aodo/AppData/Local/Temp/codex-clipboard-b218fabb-5c4c-4ecd-8986-febb475dc307.jpg")
out = Path(__file__).resolve().parent.parent / "verification"
deadline = time.monotonic() + 30
window.open_image(source)
status = [1]


def verify_state():
    if window.worker is None and window.session.image is not None:
        session = window.session
        assert session.image.calibration.source.startswith("标尺自动识别")
        assert abs(session.px_per_unit - 38.15) < .1
        assert abs(session.center[0] - 493) < 3
        assert abs(session.center[1] - 480.5) < 3
        window.grab().save(str(out / "native-reference.png"))
        save_project(out / "reference-example.saed.json", session, window.style)
        restored, _ = load_project(out / "reference-example.saed.json")
        assert len(restored.spots) == len(session.spots)
        print(f"NATIVE OK: scale={session.px_per_unit:.5f}, spots={len(session.spots)}, "
              f"sets={len(session.sets)}, project restored.")
        status[0] = 0
        window.close()
        app.quit()
    elif time.monotonic() > deadline:
        print("NATIVE TIMEOUT")
        if window.worker:
            window.worker.wait()
        window.close()
        app.quit()
    else:
        QTimer.singleShot(100, check)

def check():
    try:
        verify_state()
    except Exception:
        import traceback
        traceback.print_exc()
        if window.worker:
            window.worker.wait()
        window.close()
        app.quit()


QTimer.singleShot(100, check)
app.exec()
raise SystemExit(status[0])
