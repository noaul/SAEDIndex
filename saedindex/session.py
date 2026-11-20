"""Analysis state of one diffraction image (Qt-free)."""
from dataclasses import dataclass, field
from pathlib import Path
import csv

import numpy as np

from .calibration import Calibration, LoadedImage, load_image, nice_bar_value
from .crystal import load_cif
from .fft import (auto_zoom, compute_fft, default_roi, detect_peaks, inverse_fft, real_scale_from_axes,
                  validate_roi)
from .indexing import (Candidate, equivalent_labellings, evaluate, index_lattice,
                       measured_from_lattice, pair_candidates, simulate_zone, zone_of)
from .pattern import (Lattice, Spot, assign, detect_spots, find_center, find_lattices,
                      gauss_reduce, refine_centroid, refine_lattice, remove_line_artifacts,
                      screen_cross)

SET_COLORS = ["#ffffff", "#ffd400", "#3fd0ff", "#ff6b6b", "#8cff7a", "#d78bff"]
FFT_VERSION = "fft-1: mean removed, Hann window, fftshift; IFFT from unwindowed spectrum"
REAL_FACTORS = {"nm": 1., "\u00c5": .1, "A": .1}
# parallelogram edges as lattice nodes (n1, n2)
BASIS_CHOICES = [((1, 0), (0, 1)), ((1, 0), (1, 1)), ((0, 1), (1, 1)),
                 ((1, 0), (1, -1)), ((0, 1), (1, -1)), ((1, 1), (1, -1))]


def basis_text(pair):
    names = {(1, 0): "g1", (0, 1): "g2", (1, 1): "g1+g2", (1, -1): "g1-g2"}
    return f"{names[pair[0]]} / {names[pair[1]]}"


@dataclass
class HRTEMState:
    """Real-space source image and the processing parameters of its FFT/IFFT."""
    source: LoadedImage
    roi: tuple
    nm_per_px: float | None = None
    scale_source: str = "未标定"
    scale_confident: bool = False
    anisotropic: bool = False
    metadata_nm_per_px: float | None = None
    label_nm_per_px: float | None = None
    reference_points: list | None = None
    reference_value: float | None = None
    reference_unit: str = "nm"
    ifft_points: list | None = None      # FFT pixel coordinates; None = all detected peaks
    ifft_radius: float = 4.
    ifft_soft: float = 2.
    real_bar_value: float | None = None
    zoom_setting: int = 0                # 0 = automatic
    fft_zoom: int = 1
    fft: object = None
    ifft: np.ndarray | None = None

    PARAMS = ("roi", "nm_per_px", "scale_source", "scale_confident", "anisotropic",
              "metadata_nm_per_px", "label_nm_per_px", "reference_points", "reference_value",
              "reference_unit", "ifft_points", "ifft_radius", "ifft_soft", "real_bar_value",
              "zoom_setting", "fft_zoom")

    def choose_zoom(self):
        self.fft_zoom = self.zoom_setting or auto_zoom(self.source.raw, self.roi)

    def params(self):
        data = {k: getattr(self, k) for k in self.PARAMS}
        data["roi"] = [int(v) for v in self.roi]
        return data


def hrtem_scale(image):
    """Real-space scale evidence of a loaded HRTEM image: (nm/px, source, confident, notes)."""
    meta, cal = image.meta, image.calibration
    notes, metadata, anisotropic = [], None, False
    axes = meta.get("axis_scales")
    shape = meta.get("dm_shape")
    if axes and (shape is None or tuple(shape) == image.raw.shape):
        found = real_scale_from_axes(axes)
        if found:
            metadata, anisotropic = found
    elif axes:
        notes.append("图片与同名 DM 尺寸不同，未沿用 DM 比例")
    label = None
    if cal.bar and cal.label_value and cal.label_unit == "nm":
        label = cal.label_value / cal.bar["pixels"]
    if anisotropic:
        notes.append("DM 两轴像素比例不同，已禁止 FFT 绝对标定")
        return None, "DM 像素各向异性", False, anisotropic, metadata, label, notes
    if metadata:
        if label and abs(label / metadata - 1) > .03:
            notes.append(f"DM {metadata:.5g} 与图中标尺 {label:.5g} nm/px 相差超过 3%，请人工核对")
            return None, "DM 与图中标尺冲突，待人工核对", False, False, metadata, label, notes
        return metadata, "DM 实空间元数据", True, False, metadata, label, notes
    if label:
        confident = cal.label_confidence > .55
        if not confident:
            notes.append("标尺数字识别置信度偏低，请核对后应用")
        return (label if confident else None,
                f"实空间标尺{'自动识别' if confident else '待核对'} ({cal.label_text})",
                confident, False, metadata, label, notes)
    if cal.label_value and cal.label_unit.startswith("1/"):
        notes.append("图中为倒易标尺，不能用于 HRTEM 实空间定标")
    notes.append("未找到实空间比例，请在原图上用标尺两端定标")
    return None, "未标定", False, False, metadata, label, notes


@dataclass
class SpotSet:
    lattice: Lattice
    color: str
    name: str
    candidates: list = field(default_factory=list)
    choice: int = 0
    labelling: int = 0
    basis: int = 0
    visible: bool = True
    show_sim: bool = False
    title: str = ""                 # text after the zone axis on the figure (e.g. phase symbol)
    label_offsets: dict = field(default_factory=dict)   # key -> (dx, dy) in image px
    custom: Candidate | None = None
    confirmed: bool = False

    @property
    def candidate(self):
        if self.custom is not None:
            return self.custom
        if 0 <= self.choice < len(self.candidates):
            return self.candidates[self.choice]
        return None

    def labellings(self):
        c = self.candidate
        if c is None:
            return []
        return equivalent_labellings(c.phase, c.h1, c.h2, 1)

    def indices(self):
        """(h1, h2, zone) of the current labelling, or None."""
        c = self.candidate
        if c is None:
            return None
        options = self.labellings()
        h1, h2 = options[self.labelling] if 0 <= self.labelling < len(options) else (c.h1, c.h2)
        return np.asarray(h1, int), np.asarray(h2, int), zone_of(h1, h2)

    def hkl_of(self, node):
        found = self.indices()
        if found is None:
            return None
        h1, h2, _ = found
        return tuple(int(v) for v in node[0] * h1 + node[1] * h2)


class Session:
    def __init__(self):
        self.phases = []
        self.image = None
        self.px_per_unit = None
        self.spots = []
        self.center = None
        self.sets = []
        self.sensitivity = 6.0
        self.min_distance = 6
        self.min_diameter = 0.
        self.max_diameter = 0.
        self.smoothing = 1.2
        self.max_sets = 3
        self.length_tol = 0.03
        self.scale_tol = 0.06
        self.angle_tol = 3.0
        self.include_forbidden = False
        self.bar_value = None
        self.bar_unit = "1/nm"
        self.grouping_report = []
        self.mode = "saed"
        self.hrtem = None

    @property
    def geometry_only(self):
        return self.mode == "hrtem"

    # ------------------------------------------------------------- loading
    def open(self, path):
        self.mode, self.hrtem = "saed", None
        self.image = load_image(path)
        cal = self.image.calibration
        self.px_per_unit = cal.px_per_unit
        self.spots, self.sets, self.center = [], [], None
        self.grouping_report = []
        value = cal.label_value if cal.label_value and cal.label_unit.startswith("1/") else None
        if self.px_per_unit:
            self.bar_value = value or nice_bar_value(self.px_per_unit, self.image.gray.shape[1])
        else:
            self.bar_value = value or 5
        return self.image

    def open_hrtem(self, path):
        """Real-space lattice image; indexing runs on the FFT of a square ROI."""
        source = load_image(path)
        exclude = source.calibration.label_rect
        state = HRTEMState(source, default_roi(source.gray.shape, exclude))
        (state.nm_per_px, state.scale_source, state.scale_confident, state.anisotropic,
         state.metadata_nm_per_px, state.label_nm_per_px, notes) = hrtem_scale(source)
        self.mode, self.hrtem = "hrtem", state
        self._hrtem_notes = notes
        state.choose_zoom()
        self.rebuild_fft()
        return self.image

    def rebuild_fft(self):
        """Recompute the FFT image of the current ROI; clears spots and sets."""
        state = self.hrtem
        scale = None if state.anisotropic else state.nm_per_px
        result = compute_fft(state.source.raw, state.roi, scale, state.fft_zoom)
        state.fft, state.ifft = result, None
        cal = Calibration(px_per_unit=result.px_per_unit,
                          source=(f"{state.scale_source} · {state.nm_per_px:.5g} nm/px · "
                                  f"ROI {result.size} px · {result.zoom}×" if result.px_per_unit
                                  else state.scale_source),
                          confident=bool(result.px_per_unit and state.scale_confident),
                          notes=list(getattr(self, "_hrtem_notes", [])))
        meta = dict(state.source.meta, view="fft", roi=list(state.roi))
        self.image = LoadedImage(state.source.path, result.magnitude, result.display(), cal, meta)
        self.px_per_unit = result.px_per_unit
        self.center = result.center.copy()
        self.spots, self.sets, self.grouping_report = [], [], []
        if self.px_per_unit:
            self.bar_value = nice_bar_value(self.px_per_unit, result.size)
        if state.nm_per_px and not state.real_bar_value:
            state.real_bar_value = nice_bar_value(1 / state.nm_per_px, state.source.gray.shape[1])

    def set_roi(self, roi, analyse=True):
        if self.mode != "hrtem":
            raise ValueError("ROI 仅用于 HRTEM 图像。")
        roi = validate_roi(self.hrtem.source.raw.shape, roi)
        self.hrtem.roi = roi
        self.hrtem.ifft_points = None
        self.hrtem.choose_zoom()
        self.rebuild_fft()
        if analyse:
            self.analyse()

    def set_fft_zoom(self, setting, analyse=True):
        """0 = automatic; otherwise 1, 2, 4 or 8 times finer FFT sampling around zero."""
        if self.mode != "hrtem":
            raise ValueError("FFT 倍率仅用于 HRTEM 图像。")
        if setting not in (0, 1, 2, 4, 8):
            raise ValueError("FFT 显示倍率须为自动、1、2、4 或 8。")
        self.hrtem.zoom_setting = setting
        self.hrtem.ifft_points = None
        self.hrtem.choose_zoom()
        self.rebuild_fft()
        if analyse:
            self.analyse()

    def set_real_scale(self, nm_per_px, source="人工输入"):
        """Commit a reviewed real-space scale; FFT peak positions do not move."""
        if self.mode != "hrtem":
            raise ValueError("实空间比例仅用于 HRTEM 图像。")
        if nm_per_px is None or not np.isfinite(nm_per_px) or nm_per_px <= 0:
            raise ValueError("实空间比例须为正数 nm/px。")
        state = self.hrtem
        state.nm_per_px, state.scale_source, state.scale_confident = float(nm_per_px), source, True
        state.anisotropic = False
        state.real_bar_value = nice_bar_value(1 / state.nm_per_px, state.source.gray.shape[1])
        spots, sets, center = self.spots, self.sets, self.center
        self._hrtem_notes = []
        self.rebuild_fft()
        self.spots, self.sets, self.center = spots, sets, center
        self.index_all()

    def apply_real_reference(self, points, value, unit):
        """Two points on the source image spanning a known real-space length."""
        points = np.asarray(points, float)
        h, w = self.hrtem.source.gray.shape
        factor = REAL_FACTORS.get(unit)
        if points.shape != (2, 2) or not np.isfinite(points).all() or (points < 0).any() \
                or (points[:, 0] > w - 1).any() or (points[:, 1] > h - 1).any():
            raise ValueError("标尺端点必须位于原图内。")
        length = float(np.linalg.norm(points[1] - points[0]))
        if length <= 2 or not np.isfinite(value) or value <= 0 or factor is None:
            raise ValueError("标尺长度须大于 2 px，数值须为正数，单位须为 nm 或 Å。")
        self.set_real_scale(value * factor / length, "人工核对实空间标尺")
        state = self.hrtem
        state.reference_points, state.reference_value, state.reference_unit = \
            points.tolist(), float(value), unit

    def hrtem_params(self):
        return self.hrtem.params() if self.hrtem else None

    def restore_hrtem_params(self, params):
        """Restore processing parameters; rebuilds the FFT only when its inputs changed."""
        state = self.hrtem
        rebuild = (tuple(params["roi"]) != tuple(state.roi) or params["nm_per_px"] != state.nm_per_px
                   or params.get("fft_zoom", 1) != state.fft_zoom
                   or params["anisotropic"] != state.anisotropic or state.fft is None)
        for key, value in params.items():
            setattr(state, key, tuple(value) if key == "roi" else value)
        if rebuild:
            self.rebuild_fft()
        else:
            state.ifft = None

    def ifft_peaks(self):
        if self.hrtem is None:
            return []
        if self.hrtem.ifft_points is not None:
            return [tuple(p) for p in self.hrtem.ifft_points]
        return [tuple(p.xy) for p in self.spots]

    def toggle_ifft_spot(self, index):
        points = self.ifft_peaks()
        xy = tuple(self.spots[index].xy)
        c = self.hrtem.fft.center
        partner = tuple(2 * c - np.asarray(xy))
        near = [p for p in points if min(np.hypot(*(np.subtract(p, xy))),
                                         np.hypot(*(np.subtract(p, partner)))) < 1.5]
        if near:
            points = [p for p in points if p not in near]
        else:
            points.append(xy)
        self.hrtem.ifft_points = points
        self.hrtem.ifft = None
        return not near

    def compute_ifft(self):
        state = self.hrtem
        state.ifft = inverse_fft(state.fft, self.ifft_peaks(), state.ifft_radius, state.ifft_soft)
        return state.ifft

    def roi_crop(self):
        x, y, n = self.hrtem.roi
        return self.hrtem.source.raw[y:y + n, x:x + n]

    def add_phase(self, path):
        for phase in self.phases:
            if Path(phase.path) == Path(path):
                return phase
        phase = load_cif(path)
        self.phases.append(phase)
        return phase

    @property
    def exclude(self):
        cal = self.image.calibration if self.image else None
        return [cal.label_rect] if cal and cal.label_rect else []

    # ------------------------------------------------------------ analysis
    def analyse(self):
        """Spots, centre, spot sets and candidates in one go."""
        self.detect()
        self.find_sets()
        self.index_all()

    def detect(self):
        self.grouping_report = []
        if self.mode == "hrtem":
            if self.hrtem.fft is None:
                self.rebuild_fft()
            self.spots = detect_peaks(self.hrtem.fft, self.sensitivity)
            self.center = self.hrtem.fft.center.copy()
            self.sets = []
            return
        self.spots = detect_spots(self.image.raw, self.sensitivity, self.min_distance, self.exclude,
                                 smoothing=self.smoothing, min_diameter=self.min_diameter,
                                 max_diameter=self.max_diameter)
        self.center = find_center(self.image.raw, self.spots, self.exclude)
        self.sets = []

    def find_sets(self):
        self.grouping_report = []
        if self.mode == "hrtem":
            radius = max(3., self.hrtem.fft.size / 120)
            lattices = find_lattices(self.spots, self.center, self.max_sets, min_spots=4,
                                     min_length=radius, min_occupancy=.05, min_radius=radius,
                                     fixed_center=True, report=self.grouping_report)
        else:
            lattices = find_lattices(self.spots, self.center, self.max_sets,
                                     report=self.grouping_report)
        self.sets = []
        for lattice in lattices:
            self._append_set(lattice)
        if lattices and self.mode != "hrtem":
            self.center = lattices[0].center
            for s in self.sets[1:]:
                s.lattice.center = self.center
                self.reassign(s)

    def _append_set(self, lattice):
        k = len(self.sets)
        names = {s.name for s in self.sets}
        number = next(i for i in range(1, len(names) + 2) if f"组 {i}" not in names)
        color = next((c for c in SET_COLORS if all(s.color != c for s in self.sets)),
                     SET_COLORS[k % len(SET_COLORS)])
        s = SpotSet(lattice, color, f"组 {number}")
        self.sets.append(s)
        return s

    def measured(self, s):
        return measured_from_lattice(s.lattice, self.spots, self.px_per_unit)

    def index_set(self, s):
        s.custom = None
        s.choice, s.labelling, s.confirmed = 0, 0, False
        if not self.phases or not self.px_per_unit:
            s.candidates = []
            return
        active = [p for p in self.phases if getattr(p, "enabled", True)]
        s.candidates = index_lattice(active, self.measured(s), self.length_tol, self.angle_tol,
                                     self.scale_tol, self.include_forbidden,
                                     geometry_only=self.geometry_only)

    def index_all(self):
        for s in self.sets:
            self.index_set(s)

    def set_custom(self, s, h1, h2):
        """User-typed indices for g1 and g2; evaluated against the same phase."""
        if not np.any(np.cross(h1, h2)):
            raise ValueError("晶面指数必须为非零且不共线的两组整数。")
        phase = (s.candidate.phase if s.candidate else self.phases[0])
        c = evaluate(phase, h1, h2, self.measured(s), self.length_tol, self.angle_tol, self.scale_tol,
                     self.geometry_only)
        s.custom, s.labelling = c, 0
        options = s.labellings()
        for i, (a, b) in enumerate(options):
            if tuple(a) == tuple(h1) and tuple(b) == tuple(h2):
                s.labelling = i
                break
        return c

    # ------------------------------------------------------- manual editing
    def reassign(self, s):
        lattice = s.lattice
        if not self.spots:
            lattice.members = {}
            return
        pts = np.array([[p.x, p.y] for p in self.spots])
        n, residual, ok = assign(pts, lattice.center, lattice.g1, lattice.g2)
        ok &= np.linalg.norm(pts - lattice.center, axis=1) > 3
        lattice.members = {int(i): (int(n[i, 0]), int(n[i, 1])) for i in np.flatnonzero(ok)}
        lattice.rms = float(np.sqrt(np.mean(residual[ok] ** 2))) if ok.any() else 0.0

    def refine(self, s):
        lattice = refine_lattice(self.spots, s.lattice.center, s.lattice.g1, s.lattice.g2,
                                 refine_center_too=False)
        s.lattice.g1, s.lattice.g2 = lattice.g1, lattice.g2
        s.lattice.members, s.lattice.rms = lattice.members, lattice.rms

    def add_spot(self, x, y):
        self.grouping_report = []
        data = self.image.raw
        spot = None
        for r in (5, 9, 16):
            spot = refine_centroid(data, int(round(x)), int(round(y)), radius=r)
            if spot is not None and np.hypot(spot.x - x, spot.y - y) < r:
                break
            spot = None
        spot = spot or Spot(float(x), float(y), 1.0)
        spot.manual = True
        self.spots.append(spot)
        for s in self.sets:
            self.reassign(s)
        return len(self.spots) - 1

    def remove_spot(self, index):
        self.grouping_report = []
        del self.spots[index]
        for s in self.sets:
            self.reassign(s)

    def nearest_spot(self, x, y, radius):
        if not self.spots:
            return None
        pts = np.array([[p.x, p.y] for p in self.spots])
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        i = int(np.argmin(d))
        return i if d[i] <= radius else None

    def set_center(self, x, y):
        if self.mode == "hrtem":
            raise ValueError("FFT 中心固定为零频，不能移动。")
        self.grouping_report = []
        # snap onto the transmitted beam centroid
        spot = refine_centroid(self.image.raw, int(round(x)), int(round(y)), radius=12)
        self.center = np.array([spot.x, spot.y]) if spot and np.hypot(spot.x - x, spot.y - y) < 12 \
            else np.array([x, y], float)
        for s in self.sets:
            s.lattice.center = self.center.copy()
            self.reassign(s)

    def set_from_spots(self, i, j):
        """New spot set from two clicked reflections (vectors from the centre)."""
        if self.center is None:
            raise ValueError("请先设定透射中心，再选择两个斑点建组。")
        if not (0 <= i < len(self.spots) and 0 <= j < len(self.spots)):
            raise ValueError("斑点已改变，请重新选择。")
        a = self.spots[i].xy - self.center
        b = self.spots[j].xy - self.center
        if min(np.linalg.norm(a), np.linalg.norm(b)) < 2 or \
                abs(screen_cross(a, b)) < 0.1 * np.linalg.norm(a) * np.linalg.norm(b):
            raise ValueError("两个斑点与中心共线，请选不共线的两个斑点。")
        if screen_cross(a, b) < 0:
            a, b = b, a
        self.grouping_report = []
        lattice = Lattice(a, b, self.center.copy())
        s = self._append_set(lattice)
        self.reassign(s)
        if len(s.lattice.members) >= 4:
            self.refine(s)
        self.index_set(s)
        return s

    def remove_set(self, s):
        for i, group in enumerate(self.sets):
            if group is s:
                del self.sets[i]
                self.grouping_report = []
                return
        raise ValueError("斑点组已不存在。")

    def apply_reference(self, points, value, unit):
        """Commit a checked reciprocal reference; figure cosmetics stay unchanged."""
        if self.mode == "hrtem":
            raise ValueError("HRTEM 请在原图上用 nm 或 Å 实空间标尺定标。")
        points = np.asarray(points, float)
        if self.image is None or points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError("请先选择有效的标尺两端。")
        h, w = self.image.gray.shape
        if (points < 0).any() or (points[:, 0] > w - 1).any() or (points[:, 1] > h - 1).any():
            raise ValueError("标尺端点必须位于图片内。")
        length = float(np.linalg.norm(points[1] - points[0]))
        factor = {"1/nm": 1., "1/A": 10., "1/\u00c5": 10.}.get(unit)
        if length <= 2 or not np.isfinite(value) or value <= 0 or factor is None:
            raise ValueError("标尺长度须大于 2 px，数值须为正数，单位须为倒易单位。")
        cal = self.image.calibration
        self.px_per_unit = cal.px_per_unit = length / (value * factor)
        cal.reference_points, cal.reference_value, cal.reference_unit = points.tolist(), value, unit
        cal.source, cal.confident = "人工核对标尺", True
        cal.conflict, cal.proposed_px_per_unit = False, None
        self.index_all()

    # --------------------------------------------------------- measurements
    def node_xy(self, s, node, measured=True):
        """Image position of a lattice node: the matching spot when present."""
        if node == (0, 0):
            return np.asarray(self.center, float)
        if measured:
            for i, n in s.lattice.members.items():
                if n == tuple(node):
                    return self.spots[i].xy
        return s.lattice.position(*node)

    def has_spot(self, s, node):
        return node == (0, 0) or tuple(node) in set(s.lattice.members.values())

    def vector_rows(self, s):
        """g1, g2, g1+g2 and g1-g2 measured from the spots versus the CIF."""
        found = s.indices()
        if self.px_per_unit is None:
            return []
        phase = s.candidate.phase if s.candidate else None
        nodes = [((1, 0), "g1"), ((0, 1), "g2"), ((1, 1), "g1+g2"), ((1, -1), "g1-g2")]
        ref = self.node_xy(s, (1, 0)) - self.center
        rows = []
        for node, name in nodes:
            v = self.node_xy(s, node) - self.center
            g = np.linalg.norm(v) / self.px_per_unit
            angle = _angle(ref, v)
            row = {"name": name, "node": node, "observed": self.has_spot(s, node),
                   "angle_observed": self.has_spot(s, node) and self.has_spot(s, (1, 0)),
                   "d_meas": 1 / g if g else float("nan"), "angle_meas": angle}
            if found is not None:
                hkl = s.hkl_of(node)
                row.update(hkl=hkl, d_cif=phase.d_spacing(hkl), kind=phase.kind_of(hkl),
                           angle_cif=phase.angle(found[0], hkl) if node != (1, 0) else 0.0)
                row["err"] = row["d_meas"] / row["d_cif"] - 1
            rows.append(row)
        # closure of the measured vectors: g3 - (g1 + g2)
        if all(self.has_spot(s, n) for n in ((1, 0), (0, 1), (1, 1))):
            g1 = self.node_xy(s, (1, 0)) - self.center
            g2 = self.node_xy(s, (0, 1)) - self.center
            g3 = self.node_xy(s, (1, 1)) - self.center
            rows[2]["closure_px"] = float(np.linalg.norm(g3 - g1 - g2))
            rows[2]["closure_rel"] = float(np.linalg.norm(g3 - g1 - g2) / np.linalg.norm(g3))
        return rows

    def spot_rows(self, s):
        """Every spot of a set with its indices and d-spacings."""
        found = s.indices()
        phase = s.candidate.phase if s.candidate else None
        rows = []
        for i, node in sorted(s.lattice.members.items(),
                              key=lambda kv: np.linalg.norm(self.spots[kv[0]].xy - self.center)):
            v = self.spots[i].xy - self.center
            g = np.linalg.norm(v) / self.px_per_unit
            row = {"spot": i, "node": node, "x": self.spots[i].x, "y": self.spots[i].y,
                   "d_meas": 1 / g, "intensity": self.spots[i].intensity}
            if found is not None:
                hkl = s.hkl_of(node)
                row.update(hkl=hkl, d_cif=phase.d_spacing(hkl), kind=phase.kind_of(hkl))
                row["err"] = row["d_meas"] / row["d_cif"] - 1
            rows.append(row)
        return rows

    def simulation(self, s, gmax=None, free_scale=False):
        """CIF reflections of the chosen zone, rotated onto the measured spots."""
        found = s.indices()
        if found is None or self.px_per_unit is None:
            return [], 0.0, 1.0
        h1, h2, zone = found
        phase = s.candidate.phase
        fit = [(s.hkl_of(n), self.spots[i].xy) for i, n in s.lattice.members.items()]
        h, w = self.image.gray.shape
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
        rmax = np.max(np.linalg.norm(corners - self.center, axis=1)) / self.px_per_unit
        reflections, theta, scale = simulate_zone(phase, zone, h1, s.lattice.g1, s.lattice.g2,
                                                  self.center, self.px_per_unit,
                                                  gmax or min(rmax, phase.gmax), fit, free_scale, True)
        return reflections, theta, scale / self.px_per_unit

    # ---------------------------------------------------------------- export
    def write_csv(self, path):
        from .diagnostics import SCORE_LABELS, group_diagnostics
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            image = self.image
            writer.writerow(["图片", str(image.path)])
            writer.writerow(["标定", f"{self.px_per_unit:.4f} px/(1/nm)", image.calibration.source])
            if self.mode == "hrtem":
                state = self.hrtem
                writer.writerow(["HRTEM 实空间比例", f"{state.nm_per_px:.6g} nm/px", state.scale_source,
                                 "ROI x/y/边长", *state.roi, FFT_VERSION])
                writer.writerow(["IFFT", "掩膜半径 px", state.ifft_radius, "软边 px", state.ifft_soft,
                                 "选点", "全部" if state.ifft_points is None else
                                 "; ".join(f"({x:.2f}, {y:.2f})" for x, y in state.ifft_points)])
            cal = image.calibration
            writer.writerow(["原始比例证据", "DM px/(1/nm)", cal.metadata_px_per_unit,
                             "图中标尺 px/(1/nm)", cal.label_px_per_unit,
                             "待核对冲突", cal.conflict])
            writer.writerow([])
            for s in self.sets:
                c = s.candidate
                found = s.indices()
                writer.writerow([s.name, c.phase.name if c else "未标定",
                                 c.phase.uvw_text(found[2]) if found else "",
                                 f"整体比例误差 {c.scale_err * 100:+.2f}%" if c else "",
                                 "已确认" if s.confirmed else "待确认"])
                writer.writerow(["矢量", "hkl", "d 实测/nm", "d CIF/nm", "误差%", "夹角实测/°",
                                 "夹角 CIF/°", "有斑点", "闭合误差 px"])
                for r in self.vector_rows(s):
                    writer.writerow([r["name"], c.phase.hkl_text(r["hkl"]) if found else "",
                                     f"{r['d_meas']:.4f}" if r["observed"] else "",
                                     f"{r['d_cif']:.4f}" if found else "",
                                     f"{r['err'] * 100:+.2f}" if found and r["observed"] else "",
                                     f"{r['angle_meas']:.2f}" if r["angle_observed"] else "",
                                     f"{r['angle_cif']:.2f}" if found else "",
                                     "是" if r["observed"] else "否",
                                     f"{r['closure_px']:.2f}" if "closure_px" in r else ""])
                writer.writerow(["斑点", "hkl", "d 实测/nm", "d CIF/nm", "误差%", "x", "y", "类型"])
                for r in self.spot_rows(s):
                    writer.writerow([r["spot"], c.phase.hkl_text(r["hkl"]) if found else str(r["node"]),
                                     f"{r['d_meas']:.4f}", f"{r['d_cif']:.4f}" if found else "",
                                     f"{r['err'] * 100:+.2f}" if found else "",
                                     f"{r['x']:.2f}", f"{r['y']:.2f}", r.get("kind", "")])
                diagnostic = group_diagnostics(self, s)
                writer.writerow(["诊断汇总", "支持点", diagnostic["support"],
                                 "相对前组新增点", diagnostic["new_support"],
                                 "共享点", diagnostic["shared_support"],
                                 "RMS px", diagnostic["rms_px"]])
                writer.writerow(["评分分项", *[f"{SCORE_LABELS.get(k, k)}={v:.5f}"
                                            for k, v in diagnostic["cost_parts"].items()]])
                writer.writerow(["斑点诊断", "归属", "晶格残差 px", "d 相对误差"])
                for r in diagnostic["spots"]:
                    writer.writerow([r["spot"], r["status"], r["residual_px"], r["d_error"]])
                for warning in diagnostic["warnings"]:
                    writer.writerow(["核对提示", warning])
                writer.writerow([])


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(a @ b / na / nb, -1, 1))))
