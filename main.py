"""SAEDIndex desktop entry point."""
import argparse
import logging
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from saedindex.window import ASSETS, MainWindow


def main():
    parser = argparse.ArgumentParser(description="SAEDIndex diffraction workbench")
    parser.add_argument("file", nargs="?")
    args = parser.parse_args()
    if sys.platform == "win32":
        # Own taskbar identity, so Windows shows the app icon instead of python.exe's.
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ResearchTools.SAEDIndex")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("SAEDIndex")
    app.setOrganizationName("ResearchTools")
    app.setWindowIcon(QIcon(str(ASSETS / "app.ico")))
    app.setFont(QFont("Microsoft YaHei UI", 9))
    log_dir = Path.home() / "AppData/Local/SAEDIndex"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / "errors.log", encoding="utf-8", level=logging.ERROR)

    def report(exc_type, error, tb):
        logging.error("Unhandled error", exc_info=(exc_type, error, tb))
        QMessageBox.critical(None, "操作未完成", f"{error}\n\n诊断日志：{log_dir}")

    sys.excepthook = report
    window = MainWindow()
    window.show()
    if args.file:
        def open_file():
            if args.file.lower().endswith(".saed.json"):
                from saedindex.project import load_project
                window.session, window.style = load_project(args.file)
                window._sync_controls()
                window.refresh(reset=True)
            else:
                window.open_image(args.file)
        QTimer.singleShot(50, open_file)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
