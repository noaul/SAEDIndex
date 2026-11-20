"""Portable JSON state referencing immutable source images and CIF files."""
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .indexing import evaluate
from .pattern import Lattice, Spot
from .rendering import FigureStyle
from .session import FFT_VERSION, Session, SpotSet

SETTINGS = ("sensitivity", "min_distance", "max_sets", "length_tol", "scale_tol",
            "angle_tol", "include_forbidden", "min_diameter", "max_diameter", "smoothing")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    stored = str(path) if path.drive.lower() != root.drive.lower() else os.path.relpath(path, root)
    return {"path": stored, "sha256": _hash(path)}


def _resolve(ref, root):
    path = (root / ref["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"源文件不存在：{path}")
    if _hash(path) != ref["sha256"]:
        raise ValueError(f"源文件已改变，不能恢复原确认结果：{path}")
    return path


def save_project(path, session, style):
    path = Path(path)
    root = path.parent
    phases = session.phases

    def candidate_data(c):
        if c is None:
            return None
        return {"phase": next(i for i, p in enumerate(phases) if p is c.phase),
                "h1": c.h1.tolist(), "h2": c.h2.tolist()}

    sets = []
    for s in session.sets:
        data = {name: getattr(s, name) for name in ("name", "color", "choice", "labelling", "basis",
                "visible", "show_sim", "title", "label_offsets", "confirmed")}
        data["lattice"] = {"g1": s.lattice.g1.tolist(), "g2": s.lattice.g2.tolist(),
                           "center": s.lattice.center.tolist(), "rms": s.lattice.rms,
                           "members": {str(k): list(v) for k, v in s.lattice.members.items()}}
        data["candidates"] = [candidate_data(c) for c in s.candidates]
        data["custom"] = candidate_data(s.custom)
        sets.append(data)
    payload = {"version": 2, "mode": session.mode, "image": _reference(session.image.path, root),
               "phases": [{**_reference(p.path, root), "enabled": getattr(p, "enabled", True)}
                          for p in phases],
               "settings": {name: getattr(session, name) for name in SETTINGS},
               "calibration": asdict(session.image.calibration),
               "px_per_unit": session.px_per_unit, "bar_value": session.bar_value,
               "center": session.center.tolist() if session.center is not None else None,
               "spots": [asdict(p) for p in session.spots], "sets": sets, "style": asdict(style)}
    payload["grouping_report"] = session.grouping_report
    if session.mode == "hrtem":
        payload["hrtem"] = {**session.hrtem_params(), "processing": FFT_VERSION}
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_project(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in (1, 2):
        raise ValueError("不支持的项目版本。")
    root = path.parent
    session = Session()
    source = _resolve(payload["image"], root)
    if payload.get("mode") == "hrtem":
        session.open_hrtem(source)
        params = {k: v for k, v in payload["hrtem"].items() if k in session.hrtem.PARAMS}
        session.restore_hrtem_params(params)
        session.rebuild_fft()
    else:
        session.open(source)
    for ref in payload["phases"]:
        p = session.add_phase(_resolve(ref, root))
        p.enabled = ref.get("enabled", True)
    for name in SETTINGS:
        if name in payload["settings"]:
            setattr(session, name, payload["settings"][name])
    session.image.calibration = Calibration(**payload["calibration"])
    session.px_per_unit = payload["px_per_unit"]
    if session.px_per_unit is not None and (not np.isfinite(session.px_per_unit) or session.px_per_unit <= 0):
        raise ValueError("项目标定值无效。")
    session.bar_value = payload["bar_value"]
    session.grouping_report = payload.get("grouping_report", [])
    session.center = np.array(payload["center"], float) if payload["center"] is not None else None
    session.spots = [Spot(**p) for p in payload["spots"]]
    for data in payload["sets"]:
        lat = data["lattice"]
        lattice = Lattice(np.array(lat["g1"], float), np.array(lat["g2"], float),
                          np.array(lat["center"], float),
                          {int(k): tuple(v) for k, v in lat["members"].items()}, lat["rms"])
        if abs(np.linalg.det(np.column_stack([lattice.g1, lattice.g2]))) < 1e-6:
            raise ValueError("项目包含退化晶格。")
        s = SpotSet(lattice, data["color"], data["name"])
        for name in ("choice", "labelling", "basis", "visible", "show_sim", "title", "confirmed"):
            setattr(s, name, data[name])
        s.label_offsets = {k: tuple(v) for k, v in data["label_offsets"].items()}
        m = session.measured(s) if session.px_per_unit else None

        def restore_candidate(c):
            if c is None or m is None:
                return None
            return evaluate(session.phases[c["phase"]], c["h1"], c["h2"], m,
                            session.length_tol, session.angle_tol, session.scale_tol,
                            session.geometry_only)

        s.candidates = [restore_candidate(c) for c in data["candidates"]]
        s.custom = restore_candidate(data["custom"])
        session.sets.append(s)
    names = {f.name for f in fields(FigureStyle)}
    style = FigureStyle(**{k: v for k, v in payload["style"].items() if k in names})
    return session, style
