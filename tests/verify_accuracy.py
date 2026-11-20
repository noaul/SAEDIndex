"""Reproducible synthetic truth and read-only real-image comparisons."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np

from fixtures.pattern_cases import cases, make_case, truth_match
from saedindex.pattern import find_lattices
from saedindex.session import Session

ROOT = Path(__file__).resolve().parent.parent


def run(output, real=False, fitter=find_lattices, stress=False):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    inputs = cases()
    if stress:
        base = np.array([[44., 3.], [10., -51.]])
        theta = np.deg2rad(17)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for seed in range(101, 111):
            for label, bases, noise in (("single", [base], 45),
                                        ("double", [base, base @ rotation.T], 45),
                                        ("noise", [], 65)):
                inputs.append(make_case(f"stress_{label}_{seed}", bases, seed=seed,
                                        missing=.35, jitter=1.1, noise=noise))
    for case in inputs:
        started = time.perf_counter()
        lattices = fitter(case.spots, case.center)
        records.append({"case": case.name, "seed": case.seed,
                        "expected_groups": case.expected_groups,
                        "detected_groups": len(lattices),
                        "elapsed_seconds": time.perf_counter() - started,
                        "center": case.center.tolist(),
                        "truth_bases": [b.tolist() for b in case.bases],
                        "groups": [{"members": len(g.members), "rms_px": g.rms,
                                    "g1": g.g1.tolist(), "g2": g.g2.tolist(),
                                    "truth_precision_recall": truth_match(case, g)}
                                   for g in lattices]})
    (output / "synthetic.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("SYNTHETIC", [(r["case"], r["detected_groups"], r["expected_groups"])
                         for r in records])
    if stress:
        print("STRESS count-correct:", sum(r["detected_groups"] == r["expected_groups"] for r in records),
              "/", len(records), flush=True)
    if not real:
        return
    manifest_path = ROOT / "verification" / "accuracy-baseline" / "real-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        with (ROOT / "verification" / "image-only-calibration.csv").open(encoding="utf-8-sig") as f:
            paths = [Path(row["file"]) for row in csv.DictReader(f)]
        series = sorted({p.parent for p in paths}, key=str)
        development = [p for p in paths if p.parent.name == "850-10000"][:4]
        held = next(p for p in series if p.name == "850-1000" or p.name == "20240308")
        selected = [(p, "development") for p in development]
        selected += [(p, "held-out-series") for p in paths if p.parent == held][:4]
        manifest = [{"path": str(p), "series": str(p.parent), "split": split,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "human_verified_groups": None, "human_verified_candidates": [],
                     "truth_status": "unreviewed; not included in accuracy claims"}
                    for p, split in selected if p.exists()]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    phases = [ROOT_CIF for name in ("Ni_mp-23", "NbNi3_mp-11513", "达尔塔NbNi3_mp-1451", "NbC_mp-910")
              if (ROOT_CIF := Path("D:/cif") / (name + "_conventional_standard.cif")).exists()]
    session = Session()
    for path in phases:
        session.add_phase(path)
    observations = []
    for entry in manifest:
        path = Path(entry["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Benchmark source changed: {path}")
        started = time.perf_counter()
        session.open(path)
        session.analyse()
        row = {**entry, "elapsed_seconds": time.perf_counter() - started,
               "px_per_unit": session.px_per_unit, "center": session.center.tolist(),
               "spots": len(session.spots), "groups": []}
        for group in session.sets:
            c = group.candidate
            row["groups"].append({"members": len(group.lattice.members), "rms_px": group.lattice.rms,
                "candidate": c.label if c else None, "cost": float(c.cost) if c else None,
                "scale_error": float(c.scale_err) if c else None,
                "angle_error_deg": float(c.angle_meas - c.angle_theo) if c else None,
                "confirmed": group.confirmed})
        observations.append(row)
        print("REAL", entry["split"], path.name, len(session.sets), flush=True)
    (output / "real.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--baseline-archive", type=Path)
    args = parser.parse_args()
    if args.baseline_archive:
        if args.real:
            parser.error("Archive comparison only supports synthetic cases")
        with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(args.baseline_archive) as archive:
            name = next(n for n in archive.namelist() if n.replace('\\', '/') == 'saedindex/pattern.py')
            path = Path(archive.extract(name, temp))
            spec = importlib.util.spec_from_file_location("saedindex_baseline_pattern", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            run(args.output, fitter=module.find_lattices, stress=args.stress)
    else:
        run(args.output, args.real, stress=args.stress)
