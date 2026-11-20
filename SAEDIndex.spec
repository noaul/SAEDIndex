# PyInstaller build: .\venv\Scripts\pyinstaller.exe --noconfirm SAEDIndex.spec
# Output: dist\SAEDIndex\SAEDIndex.exe (one-folder build, faster start than one-file).

a = Analysis(
    ["main.py"],
    pathex=["."],
    datas=[("assets", "assets")],
    hiddenimports=["PySide6.QtSvg"],
    # The venv inherits system site-packages; keep optional heavy libraries out.
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "torch", "torchvision",
              "tensorflow", "onnxruntime", "numba", "llvmlite", "sympy", "pandas",
              "sklearn", "cv2", "cryptography", "lxml", "google", "setuptools",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
              "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf"],
    noarchive=False,
)
# Only bundle binaries from the Python install or the venv. DLLs that PyInstaller
# finds through PATH (system UCRT, other tools' ICU/OpenSSL) break the QtCore
# import with "DLL load failed ... procedure not found".
import os, sys
_roots = tuple(os.path.normcase(os.path.abspath(p)) + os.sep
               for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix})
a.binaries = [b for b in a.binaries
              if os.path.normcase(os.path.abspath(b[1])).startswith(_roots)]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SAEDIndex",
    icon="assets/app.ico",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="SAEDIndex", upx=False)
