"""End-to-end real data audit; sources are read-only, all outputs in verification/."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from saedindex.calibration import load_image
from saedindex.dmfile import read_dm
from saedindex.rendering import export_figure
from saedindex.window import MainWindow

OUT = Path(__file__).resolve().parent.parent / "verification"
OUT.mkdir(exist_ok=True)
APP = QApplication.instance() or QApplication([])
ROOT = Path(r"F:\E D\整理中-第四章稳定性及力学\00-热暴露\03-TEM\850-10000")
window = MainWindow()
window.show()
results = []
for name in ("1-0010", "1-0011", "1-0013", "1-0014"):
    window.open_image(ROOT / f"{name}.jpg", background=False)
    record = {"image": name, "calibration": window.session.px_per_unit,
              "spots": len(window.session.spots), "sets": []}
    for s in window.session.sets:
        c = s.candidate
        record["sets"].append({"members": len(s.lattice.members), "rms_px": s.lattice.rms,
                              "top_candidate": c.label if c else None,
                              "cost": float(c.cost) if c else None,
                              "vectors": [
                                  {key: (float(v) if isinstance(v, np.floating) else v)
                                   for key, v in r.items()} for r in window.session.vector_rows(s)]})
    results.append(record)
    if name == "1-0013":
        assert window.session.sets[0].candidate.phase.name.startswith("达尔塔")
        assert window.session.sets[0].indices()[2] == (0, 0, 1)
        window.tabs.setCurrentIndex(1)
        for width, height in ((1440, 960), (1100, 800)):
            window.resize(width, height)
            APP.processEvents()
            window.grab().save(str(OUT / f"workbench-{width}.png"))
        export_figure(window.session, window.style, OUT / "delta-example-draft.png")
        window.session.write_csv(OUT / "delta-example.csv")
        from saedindex.project import save_project, load_project
        save_project(OUT / "delta-example.saed.json", window.session, window.style)
        restored, _ = load_project(OUT / "delta-example.saed.json")
        assert len(restored.sets) == len(window.session.sets)
        assert restored.sets[0].indices()[2] == (0, 0, 1)
    if name == "1-0014":
        window.resize(1440, 960)
        window.simulation_box.setChecked(True)
        APP.processEvents()
        window.grab().save(str(OUT / "simulation.png"))
window.close()
(OUT / "pipeline.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
if "--figures-only" in sys.argv:
    print("Real-image pipeline and desktop figures verified.")
    raise SystemExit(0)

# JPEG-only calibration, reference truth read separately from matching DM metadata.
calibration = []
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / "image_only.jpg"
    for source in sorted(Path("F:/E D").rglob("*.jpg")):
        try:
            _, scale, unit, _ = read_dm(source.with_suffix(".dm3"), load_data=False)
        except Exception:
            continue
        if unit != "1/nm" or not scale:
            continue
        shutil.copyfile(source, path)
        image = load_image(path)
        cal = image.calibration
        error = (cal.px_per_unit * scale - 1) * 100 if cal.px_per_unit else None
        calibration.append({"file": str(source), "label": cal.label_text,
                            "confident": bool(cal.confident), "error_percent": error,
                            "calibrated": cal.px_per_unit is not None})
with (OUT / "image-only-calibration.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(calibration[0]))
    writer.writeheader()
    writer.writerows(calibration)
correct = [r for r in calibration if r["calibrated"] and abs(r["error_percent"]) < 3]
rejected = [r for r in calibration if not r["calibrated"]]
wrong = [r for r in calibration if r["calibrated"] and abs(r["error_percent"]) >= 3]
summary = {"total": len(calibration), "correct_scale": len(correct),
           "rejected": len(rejected), "wrong_scale": len(wrong),
           "confident_correct": sum(r["confident"] for r in correct),
           "confident_wrong": sum(r["confident"] for r in wrong),
           "correct_mean_abs_error_percent": float(np.mean([abs(r["error_percent"]) for r in correct])),
           "correct_max_abs_error_percent": max(abs(r["error_percent"]) for r in correct)}
(OUT / "calibration-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
for r in wrong + rejected:
    print("ATTENTION", r)
