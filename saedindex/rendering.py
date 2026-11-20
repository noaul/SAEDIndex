"""Shared on-screen and publication figure rendering."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QImage, QPainter, QPen, QPolygonF

from .crystal import display_indices
from .diagnostics import simulation_matches
from .fft import to_uint8
from .session import BASIS_CHOICES

_fonts_ready = False


def ensure_fonts():
    global _fonts_ready
    if not _fonts_ready:
        for name in ("msyh.ttc", "msyhbd.ttc", "times.ttf", "timesbd.ttf"):
            path = Path("C:/Windows/Fonts") / name
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
        _fonts_ready = True


@dataclass
class FigureStyle:
    brightness: float = 0.
    contrast: float = 1.
    gamma: float = 1.
    log: bool = False
    font_size: int = 38
    font_bold: bool = False
    line_width: float = 2.5
    replace_bar: bool = True
    bar_font_size: int = 56
    bar_thickness: int = 12
    show_spots: bool = True
    show_center: bool = True
    show_annotations: bool = True
    show_simulation: bool = True
    free_scale: bool = False
    show_dynamic: bool = False
    simulation_size: float = 7.
    simulation_opacity: float = .95
    publication_preview: bool = False
    show_calibration: bool = True
    cover_rect: list | None = None
    show_residuals: bool = False
    chemical_subscripts: bool = True
    phase_subscript: bool = True


def chemical_text(text):
    """Display-only formula subscripts; keep sample IDs and Miller indices intact."""
    formula = r"(?<![A-Za-z0-9_])(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?|\((?:[A-Z][a-z]?\d*)+\)\d*)+(?![A-Za-z0-9_])"
    digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return re.sub(formula, lambda m: m.group().translate(digits), text)


def display_array(gray, style):
    x = np.asarray(gray, dtype=np.float32) / 255.
    if style.log:
        x = np.log1p(30. * x) / np.log1p(30.)
    x = np.clip((x - .5) * style.contrast + .5 + style.brightness / 255., 0, 1)
    return np.ascontiguousarray(np.round(x ** (1. / max(style.gamma, .05)) * 255).astype(np.uint8))


def qimage_from_array(array):
    a = np.ascontiguousarray(array, dtype=np.uint8)
    if a.ndim == 2:
        return QImage(a.data, a.shape[1], a.shape[0], a.strides[0],
                      QImage.Format.Format_Grayscale8).copy()
    return QImage(a.data, a.shape[1], a.shape[0], a.strides[0],
                  QImage.Format.Format_RGB888).copy()


def index_runs(values, bracket, font):
    """Glyphs with explicit overbar flags, independent of combining-font support."""
    ensure_fonts()
    close = ")" if bracket == "(" else "]"
    strings = [(bracket, False)]
    for i, value in enumerate(values):
        if i and any(abs(v) >= 10 for v in values):
            strings.append((" ", False))
        strings.append((str(abs(value)), value < 0))
    strings.append((close, False))
    fm = QFontMetricsF(font)
    return [(text, overbar, fm.horizontalAdvance(text)) for text, overbar in strings]


@dataclass
class Label:
    set_index: int
    key: str
    rect: QRectF
    origin: QPointF
    values: tuple
    bracket: str
    suffix: str
    subscript: bool
    color: str


def phase_font(font, subscript):
    result = QFont(font)
    if subscript:
        result.setPixelSize(max(1, round(font.pixelSize() * .70)))
    return result


def annotation_labels(session, style):
    ensure_fonts()
    if session.image is None or not style.show_annotations:
        return []
    h, w = session.image.gray.shape
    font = QFont("Times New Roman")
    font.setPixelSize(style.font_size)
    font.setBold(style.font_bold)
    fm = QFontMetricsF(font)
    suffix_metrics = QFontMetricsF(phase_font(font, style.phase_subscript))
    labels = []
    occupied = []
    for k, s in enumerate(session.sets):
        if not s.visible or s.indices() is None:
            continue
        phase = s.candidate.phase
        a, b = BASIS_CHOICES[s.basis]
        ab = (a[0] + b[0], a[1] + b[1])
        nodes = [("g1", a), ("g2", b), ("sum", ab)]
        for key, node in nodes + [("zone", None)]:
            if node is None:
                values = display_indices(s.indices()[2], phase.hexagonal, plane=False)
                title = s.title.strip() or phase.name
                if style.chemical_subscripts:
                    title = chemical_text(title)
                bracket = "["
                suffix = (" " + title if title else "")
                x, y = w * .68, h * .79 + k * (style.font_size * 1.5)
            else:
                values = display_indices(s.hkl_of(node), phase.hexagonal)
                bracket = "("
                suffix = ""
                xy = session.node_xy(s, node)
                direction = xy - session.center
                direction /= max(np.linalg.norm(direction), 1.)
                x = xy[0] + direction[0] * 20
                y = xy[1] + direction[1] * 22
            runs = index_runs(values, bracket, font)
            width = min(w - 10., sum(r[2] for r in runs) + suffix_metrics.horizontalAdvance(suffix) + 4.)
            height = fm.height() + 6.
            if suffix and style.phase_subscript:
                height += max(0., .24 * font.pixelSize() + suffix_metrics.descent() - fm.descent())
            if node is not None:
                if direction[0] < -.2:
                    x -= width
                elif abs(direction[0]) <= .2:
                    x -= width / 2
                if direction[1] < 0:
                    y -= height
            origin = QPointF(float(x), float(y))
            dx, dy = s.label_offsets.get(key, (0., 0.))
            x, y = x + dx, y + dy
            rect = QRectF(float(np.clip(x, 5, max(5, w - width - 5))),
                          float(np.clip(y, 5, max(5, h - height - 5))), width, height)
            if key not in s.label_offsets:
                for _ in range(10):
                    if not any(rect.intersects(r.adjusted(-3, -3, 3, 3)) for r in occupied):
                        break
                    rect.translate(0, -height - 4 if rect.top() > height + 10 else height + 4)
                    rect.moveTop(float(np.clip(rect.top(), 5, max(5, h - height - 5))))
            occupied.append(rect)
            labels.append(Label(k, key, rect, origin, tuple(values), bracket, suffix, style.phase_subscript, s.color))
    return labels


def _paint_label(painter, label, font):
    painter.setFont(font)
    painter.setPen(QPen(QColor(label.color), 1.3))
    fm = QFontMetricsF(font)
    baseline = label.rect.top() + fm.ascent() + 3
    x = label.rect.left() + 2.
    for text, overbar, advance in index_runs(label.values, label.bracket, font):
        painter.drawText(QPointF(x, baseline), text)
        if overbar:
            top = baseline - fm.capHeight() - 3.
            painter.drawLine(QPointF(x + .5, top), QPointF(x + advance - .5, top))
        x += advance
    if label.suffix:
        painter.setFont(phase_font(font, label.subscript))
        drop = .24 * font.pixelSize() if label.subscript else 0.
        painter.drawText(QPointF(x, baseline + drop), label.suffix)


def bar_geometry(session, style):
    h, w = session.image.gray.shape
    margin = max(12., w * .03)
    length = float(session.bar_value * session.px_per_unit)
    bar = QRectF(margin, h - margin - style.bar_thickness, length, style.bar_thickness)
    text = QRectF(margin, bar.top() - style.bar_font_size * 1.3,
                  max(length, style.bar_font_size * 4), style.bar_font_size * 1.3)
    return bar, text


def prepare_background(session, style, publication=True):
    array = display_array(session.image.gray, style)
    cover = None if session.mode == "hrtem" else (style.cover_rect or session.image.calibration.label_rect)
    if publication and style.replace_bar and session.px_per_unit and cover:
        x1, y1, x2, y2 = [int(v) for v in cover]
        # Source pixels are never modified. This is an explicit figure mask.
        h, w = array.shape
        neighborhood = array[max(0, y1 - 12):min(h, y2 + 12), max(0, x1 - 12):min(w, x2 + 12)]
        level = int(np.percentile(neighborhood, 35)) if neighborhood.size else 0
        array[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = level
    return qimage_from_array(array)


REAL_VIEWS = ("source", "ifft", "compare")
COMPARE_GAP = 8


def view_array(session, view, style):
    """Display array of a real-space HRTEM view (original, IFFT or ROI/IFFT pair)."""
    state = session.hrtem
    if view == "source":
        return display_array(state.source.gray, style)
    if state.ifft is None:
        session.compute_ifft()
    ifft = to_uint8(state.ifft)
    if view == "ifft":
        return display_array(ifft, style)
    crop = to_uint8(session.roi_crop())
    gap = np.full((ifft.shape[0], COMPARE_GAP), 255, np.uint8)
    return display_array(np.hstack([crop, gap, ifft]), style)


def real_bar_geometry(width, height, nm_per_px, value, style):
    margin = max(12., width * .03)
    length = float(value / nm_per_px)
    bar = QRectF(margin, height - margin - style.bar_thickness, length, style.bar_thickness)
    text = QRectF(margin, bar.top() - style.bar_font_size * 1.3,
                  max(length, style.bar_font_size * 4), style.bar_font_size * 1.3)
    return bar, text


def paint_bar(painter, bar, text_rect, text, style):
    painter.save()
    painter.fillRect(bar, QColor("white"))
    font = QFont("Times New Roman")
    font.setPixelSize(style.bar_font_size)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("white"))
    painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, text)
    painter.restore()


def paint_real_overlay(painter, session, view, style, editing=False, markers=True):
    """ROI frame while editing; nm scale bar on publication views."""
    state = session.hrtem
    if state is None:
        return
    if view == "source" and editing and markers:
        x, y, n = state.roi
        pen = QPen(QColor("#ffd400"), 2.)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x, y, n, n))
    if (not editing or style.publication_preview) and style.replace_bar \
            and state.nm_per_px and state.real_bar_value:
        width = state.roi[2] if view != "source" else state.source.gray.shape[1]
        height = state.roi[2] if view != "source" else state.source.gray.shape[0]
        bar, text = real_bar_geometry(width, height, state.nm_per_px, state.real_bar_value, style)
        paint_bar(painter, bar, text, f"{state.real_bar_value:g} nm", style)


def real_background(session, view, style, publication=True):
    array = view_array(session, view, style)
    cover = style.cover_rect or session.hrtem.source.calibration.label_rect
    if view == "source" and publication and style.replace_bar and session.hrtem.nm_per_px and cover:
        x1, y1, x2, y2 = [int(v) for v in cover]
        h, w = array.shape
        neighborhood = array[max(0, y1 - 12):min(h, y2 + 12), max(0, x1 - 12):min(w, x2 + 12)]
        array[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = \
            int(np.percentile(neighborhood, 35)) if neighborhood.size else 0
    return qimage_from_array(array)


def paint_overlay(painter, session, style, editing=False, simulations=None, selected=None,
                  markers=True, diagnostics=None):
    if session.image is None:
        return
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    markers = editing and markers
    if markers and style.show_spots:
        assigned = {}
        for k, s in enumerate(session.sets):
            if s.visible:
                for i in s.lattice.members:
                    assigned.setdefault(i, s.color)
        for i, p in enumerate(session.spots):
            pen = QPen(QColor(assigned.get(i, "#73a38f")), 1.1)
            if i == selected:
                pen = QPen(QColor("#ff8469"), 2.)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(p.x, p.y), 5.5, 5.5)
    if markers and style.show_center and session.center is not None:
        x, y = session.center
        painter.setPen(QPen(QColor("#ff8469"), 1.6))
        painter.drawLine(QPointF(x - 11, y), QPointF(x + 11, y))
        painter.drawLine(QPointF(x, y - 11), QPointF(x, y + 11))
    if markers and style.show_simulation:
        h, w = session.image.gray.shape
        screen_scale = max(abs(painter.worldTransform().m11()), .01)
        r = style.simulation_size / screen_scale
        painter.save()
        painter.setOpacity(style.simulation_opacity)
        for k, reflections in (simulations or {}).items():
            s = session.sets[k]
            if not s.visible or not s.show_sim:
                continue
            if style.show_residuals:
                matches = (diagnostics or {}).get(k)
                if matches is None:
                    fixed = session.simulation(s, free_scale=False)[0]
                    matches = simulation_matches(session, s, fixed)
            else:
                matches = [{"reflection": reflection} for reflection in reflections]
            for match in matches:
                hkl, x, y, intensity, allowed = match["reflection"]
                if not (0 <= x <= w and 0 <= y <= h) or (not allowed and not style.show_dynamic):
                    continue
                color = "#41efff"
                if style.show_residuals:
                    color = ("#41efff" if match["matched"] else "#ff8469"
                             if match["spot"] is not None else "#9aa9a8")
                    if match["spot"] is not None:
                        spot = session.spots[match["spot"]]
                        residual_pen = QPen(QColor(color), 1.5)
                        residual_pen.setCosmetic(True)
                        painter.setPen(residual_pen)
                        painter.drawLine(QPointF(x, y), QPointF(spot.x, spot.y))
                pen = QPen(QColor(color), 2.)
                pen.setCosmetic(True)
                if not allowed:
                    pen.setStyle(Qt.PenStyle.DotLine)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                halo = QPen(QColor("#000000"), 4.)
                halo.setCosmetic(True)
                painter.setPen(halo)
                painter.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
                painter.setPen(pen)
                painter.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
                painter.drawLine(QPointF(x - r * .35, y), QPointF(x + r * .35, y))
                painter.drawLine(QPointF(x, y - r * .35), QPointF(x, y + r * .35))
        painter.restore()
    if style.show_annotations:
        for s in session.sets:
            if not s.visible or (not editing and s.candidate is None):
                continue
            a, b = BASIS_CHOICES[s.basis]
            ab = (a[0] + b[0], a[1] + b[1])
            points = [session.node_xy(s, n) for n in ((0, 0), a, ab, b)]
            painter.setPen(QPen(QColor(s.color), style.line_width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QPolygonF([QPointF(*xy) for xy in points]))
        font = QFont("Times New Roman")
        font.setPixelSize(style.font_size)
        font.setBold(style.font_bold)
        for label in annotation_labels(session, style):
            _paint_label(painter, label, font)
    if (not editing or style.publication_preview) and style.replace_bar \
            and session.px_per_unit and session.bar_value:
        bar, text_rect = bar_geometry(session, style)
        paint_bar(painter, bar, text_rect, f"{session.bar_value:g} 1/nm", style)
    painter.restore()


def _save_qimage(image, path, dpi):
    array = np.asarray(image.constBits()).reshape(image.height(), image.bytesPerLine())
    rgb = array[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()
    output = Image.fromarray(rgb)
    options = {"dpi": (dpi, dpi)}
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        options["compression"] = "tiff_lzw"
    output.save(str(path), **options)


def export_figure(session, style, path, dpi=600, view="pattern"):
    ensure_fonts()
    if view in REAL_VIEWS:
        image = real_background(session, view, style).convertToFormat(QImage.Format.Format_RGB888)
        painter = QPainter(image)
        paint_real_overlay(painter, session, view, style, editing=False)
        painter.end()
        _save_qimage(image, path, dpi)
        return
    image = prepare_background(session, style).convertToFormat(QImage.Format.Format_RGB888)
    painter = QPainter(image)
    paint_overlay(painter, session, style, editing=False)
    painter.end()
    _save_qimage(image, path, dpi)
