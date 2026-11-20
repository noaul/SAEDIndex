"""Image canvas: display-only transforms and direct scientific annotation editing."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .rendering import (REAL_VIEWS, annotation_labels, paint_overlay, paint_real_overlay,
                        prepare_background, real_background)
from .diagnostics import simulation_matches
from .session import BASIS_CHOICES

PATTERN_MODES = ("inspect", "center", "add", "delete", "move", "pair", "delete_group", "labels",
                 "ifft_pick")
SOURCE_MODES = ("roi",)


class PatternCanvas(QWidget):
    changed = Signal()
    edit_started = Signal()
    labels_moved = Signal()
    pair_selected = Signal(int, int)
    scale_selected = Signal(float)
    reference_selected = Signal(object)
    group_delete_requested = Signal(int)
    mask_selected = Signal(object)
    inspected = Signal(int)
    message = Signal(str)
    roi_selected = Signal(object)
    ifft_toggled = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self.setMouseTracking(True)
        self.session = None
        self.style = None
        self.background = None
        self.simulations = {}
        self.simulation_diagnostics = {}
        self.mode = "pan"
        self.selected = None
        self.zoom = 1.
        self.pan = QPointF()
        self._drag = None
        self._first = None
        self._last = QPointF()
        self._mask = None
        self.active_set = 0
        self.calibration_points = None
        self.show_markers = True
        self.view = "pattern"
        self._roi = None

    @property
    def real_view(self):
        return bool(self.session and self.session.mode == "hrtem" and self.view in REAL_VIEWS)

    @property
    def calibration_view(self):
        """View on which scale endpoints live: the original image."""
        if self.session and self.session.mode == "hrtem":
            return self.view == "source"
        return self.view == "pattern"

    def bind(self, session, style, reset=False):
        self.session, self.style = session, style
        if reset:
            self.zoom, self.pan = 1., QPointF()
            self._first, self.selected = None, None
            self._drag, self._mask = None, None
        if self.selected is not None and self.selected >= len(session.spots):
            self.selected = None
        if self.mode == "pair" and self._first is not None and self._first >= len(session.spots):
            self._first = None
        self.refresh_background()
        self.refresh_simulations()

    def refresh_background(self):
        if not (self.session and self.session.image):
            self.background = None
        elif self.real_view:
            self.background = real_background(self.session, self.view, self.style,
                                              publication=self.style.publication_preview)
        else:
            self.background = prepare_background(self.session, self.style,
                                                 publication=self.style.publication_preview)
        self.update()

    def set_view(self, view):
        if self.session is None or self.session.mode != "hrtem":
            view = "pattern"
        if view != self.view:
            self.view = view
            self._first, self._drag, self._mask, self._roi = None, None, None, None
            self.zoom, self.pan = 1., QPointF()
        self.refresh_background()

    def refresh_simulations(self):
        self.simulations = {}
        self.simulation_diagnostics = {}
        if self.session and self.session.image:
            for k, s in enumerate(self.session.sets):
                if s.show_sim:
                    self.simulations[k] = self.session.simulation(
                        s, free_scale=self.style.free_scale and not self.style.show_residuals)[0]
                    if self.style.show_residuals:
                        self.simulation_diagnostics[k] = simulation_matches(self.session, s, self.simulations[k])
        self.update()

    def fit(self):
        self.zoom, self.pan = 1., QPointF()
        self.update()

    def set_mode(self, mode):
        self.mode, self._first, self._drag, self._mask, self._roi = mode, None, None, None, None
        self.setCursor(Qt.CursorShape.OpenHandCursor if mode == "pan" else Qt.CursorShape.CrossCursor)
        self.update()

    def transform_values(self):
        if self.background is None:
            return 1., QPointF()
        w, h = self.background.width(), self.background.height()
        scale = min((self.width() - 24) / w, (self.height() - 24) / h) * self.zoom
        offset = QPointF((self.width() - w * scale) / 2,
                         (self.height() - h * scale) / 2) + self.pan
        return scale, offset

    def to_widget(self, x, y):
        scale, offset = self.transform_values()
        return QPointF(x * scale, y * scale) + offset

    def to_image(self, point):
        scale, offset = self.transform_values()
        return (point - offset) / scale

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eef1f1"))
        if self.background is None:
            painter.setPen(QColor("#9aa5a4"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "SAEDIndex")
            return
        scale, offset = self.transform_values()
        painter.translate(offset)
        painter.scale(scale, scale)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QPointF(), self.background)
        painter.setClipRect(QRectF(0, 0, self.background.width(), self.background.height()))
        editing = not self.style.publication_preview
        if self.real_view:
            paint_real_overlay(painter, self.session, self.view, self.style, editing=editing,
                               markers=self.show_markers)
        else:
            paint_overlay(painter, self.session, self.style, editing=editing,
                          simulations=self.simulations, selected=self.selected,
                          markers=self.show_markers, diagnostics=self.simulation_diagnostics)
            if self.session.mode == "hrtem" and editing and self.show_markers:
                state = self.session.hrtem
                pen = QPen(QColor("#ff5fd2"), 1.6)
                pen.setCosmetic(True)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                c = state.fft.center
                r = (state.ifft_radius + state.ifft_soft / 2) * state.fft.zoom
                for x, y in self.session.ifft_peaks():
                    for px, py in ((x, y), (2 * c[0] - x, 2 * c[1] - y)):
                        painter.drawEllipse(QPointF(px, py), r, r)
        if self.calibration_points is not None and self.style.show_calibration and self.calibration_view \
                and self.show_markers and not self.style.publication_preview:
            points = [QPointF(*xy) for xy in self.calibration_points]
            pen = QPen(QColor("#ff8469"), 2.)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(*points)
            for point in points:
                painter.drawEllipse(point, 5 / scale, 5 / scale)
        if self._first is not None:
            painter.setPen(QPen(QColor("#ff8469"), 2 / scale))
            if self.mode == "pair":
                p = self.session.spots[self._first]
                painter.drawEllipse(QPointF(p.x, p.y), 9 / scale, 9 / scale)
            else:
                painter.drawEllipse(self._first, 4 / scale, 4 / scale)
        if self._mask is not None:
            painter.setPen(QPen(QColor("#ff8469"), 1.5 / scale, Qt.PenStyle.DashLine))
            painter.drawRect(self._mask)
        if self._roi is not None:
            pen = QPen(QColor("#ffd400"), 2.)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(self._roi)
        painter.end()

    def _square(self, start, p):
        """Square ROI anchored at the press point, kept inside the source image."""
        w, h = self.background.width(), self.background.height()
        dx, dy = p.x() - start.x(), p.y() - start.y()
        sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
        limit_x = (w - start.x()) if sx > 0 else start.x()
        limit_y = (h - start.y()) if sy > 0 else start.y()
        size = min(max(abs(dx), abs(dy)), limit_x, limit_y)
        x = start.x() if sx > 0 else start.x() - size
        y = start.y() if sy > 0 else start.y() - size
        return QRectF(x, y, size, size)

    def mousePressEvent(self, event):
        if self.background is None:
            return
        self._last = event.position()
        p = self.to_image(event.position())
        x, y = p.x(), p.y()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag = ("pan",)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not (0 <= x < self.background.width() and 0 <= y < self.background.height()):
            return
        scale, _ = self.transform_values()
        radius = max(7., 10 / scale)
        i = self.session.nearest_spot(x, y, radius)
        if self.session.mode == "hrtem":
            if self.view != "pattern" and self.mode in PATTERN_MODES:
                self.message.emit("该工具作用于 FFT 视图，请切换到 FFT。")
                return
            if self.view != "source" and self.mode in SOURCE_MODES + ("scale", "mask"):
                self.message.emit("该工具作用于原图视图，请切换到原图。")
                return
            if self.view in REAL_VIEWS and self.mode == "pan":
                self._drag = ("pan",)
                return
            if self.mode == "roi":
                self._drag = ("roi", p)
                self._roi = QRectF(p, p)
                self.update()
                return
            if self.mode == "center":
                self.message.emit("FFT 中心固定为零频。")
                return
            if self.mode == "ifft_pick":
                if i is None:
                    self.message.emit("未选中 FFT 斑点。")
                else:
                    self.ifft_toggled.emit(i)
                return
        if self.mode in ("labels", "pan"):
            for label in reversed(annotation_labels(self.session, self.style)):
                if label.rect.contains(p):
                    self.edit_started.emit()
                    offsets = self.session.sets[label.set_index].label_offsets
                    self._drag = ("label", label.set_index, label.key,
                                  p, label.rect.topLeft() - label.origin, offsets.get(label.key))
                    return
            if self.mode == "pan":
                self._drag = ("pan",)
        elif self.mode == "center":
            self.edit_started.emit()
            self.session.set_center(x, y)
            self.changed.emit()
        elif self.mode == "add":
            self.edit_started.emit()
            self.selected = self.session.add_spot(x, y)
            self.changed.emit()
        elif self.mode == "delete" and i is not None:
            self.edit_started.emit()
            self.session.remove_spot(i)
            self.selected = None
            self.changed.emit()
        elif self.mode in ("move", "inspect") and i is not None:
            self.selected = i
            self.inspected.emit(i)
            if self.mode == "move":
                self.edit_started.emit()
                spot = self.session.spots[i]
                self._drag = ("spot", i, QPointF(spot.x, spot.y) - p, (spot.x, spot.y))
        elif self.mode == "pair":
            if self.session.center is None:
                self.message.emit("请先设定透射中心。")
            elif i is None:
                self.message.emit("未选中斑点，可先补点。")
            elif self._first is None:
                self._first = i
                self.message.emit("已选择第一个矢量，请选第二个不共线的斑点。")
            elif i != self._first:
                first, self._first = self._first, None
                self.pair_selected.emit(first, i)
        elif self.mode == "delete_group":
            matches = []
            for k, s in enumerate(self.session.sets):
                if not s.visible:
                    continue
                a, b = BASIS_CHOICES[s.basis]
                ab = (a[0] + b[0], a[1] + b[1])
                polygon = QPolygonF([QPointF(*self.session.node_xy(s, n))
                                     for n in ((0, 0), a, ab, b)])
                if (i is not None and i in s.lattice.members) or \
                        polygon.containsPoint(p, Qt.FillRule.OddEvenFill):
                    matches.append(k)
            if matches:
                self.group_delete_requested.emit(
                    self.active_set if self.active_set in matches else matches[-1])
            else:
                self.message.emit("未选中斑点组。")
        elif self.mode == "scale":
            if self._first is None and self.calibration_points is not None:
                for k, xy in enumerate(self.calibration_points):
                    if np.linalg.norm(np.asarray(xy) - [x, y]) <= radius:
                        self._drag = ("reference", k)
                        return
            if self._first is None:
                self._first = p
                self.message.emit("已选择标尺起点，请选标尺终点。")
            else:
                length = np.hypot(p.x() - self._first.x(), p.y() - self._first.y())
                points = [[self._first.x(), self._first.y()], [p.x(), p.y()]]
                self._first = None
                if length > 2:
                    self.calibration_points = points
                    self.reference_selected.emit(points)
                    self.scale_selected.emit(length)
        elif self.mode == "mask":
            self._drag = ("mask", p)
            self._mask = QRectF(p, p)
        self.update()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        p = self.to_image(event.position())
        kind = self._drag[0]
        if kind == "pan":
            self.pan += event.position() - self._last
        elif kind == "spot":
            i = self._drag[1]
            spot = self.session.spots[i]
            target = p + self._drag[2]
            x = float(np.clip(target.x(), 0, self.background.width() - 1))
            y = float(np.clip(target.y(), 0, self.background.height() - 1))
            if np.hypot(x - self._drag[3][0], y - self._drag[3][1]) >= .5 or spot.manual:
                spot.x, spot.y, spot.manual = x, y, True
        elif kind == "label":
            _, k, key, start, offset, _ = self._drag
            d = offset + p - start
            self.session.sets[k].label_offsets[key] = (d.x(), d.y())
        elif kind == "mask":
            self._mask = QRectF(self._drag[1], p).normalized()
        elif kind == "roi":
            self._roi = self._square(self._drag[1], p)
        elif kind == "reference":
            self.calibration_points[self._drag[1]] = [
                float(np.clip(p.x(), 0, self.background.width() - 1)),
                float(np.clip(p.y(), 0, self.background.height() - 1))]
            self.reference_selected.emit([list(xy) for xy in self.calibration_points])
        self._last = event.position()
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag:
            self.mouseMoveEvent(event)
            kind = self._drag[0]
            if kind == "spot":
                spot = self.session.spots[self._drag[1]]
                if (spot.x, spot.y) != self._drag[3]:
                    for s in self.session.sets:
                        self.session.reassign(s)
                    self.changed.emit()
            elif kind == "label":
                _, k, key, *_, before = self._drag
                if self.session.sets[k].label_offsets.get(key) != before:
                    self.labels_moved.emit()
            elif kind == "mask":
                r = self._mask
                if r.width() > 3 and r.height() > 3:
                    self.mask_selected.emit([r.left(), r.top(), r.right(), r.bottom()])
                self._mask = None
            elif kind == "roi":
                r = self._roi
                self._roi = None
                if r is not None and r.width() >= 32:
                    self.roi_selected.emit([int(r.left()), int(r.top()),
                                            int(r.width())])
                else:
                    self.message.emit("ROI 边长不少于 32 px。")
        self._drag = None
        self.update()

    def wheelEvent(self, event):
        if self.background is None:
            return
        before = self.to_image(event.position())
        self.zoom = float(np.clip(self.zoom * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15),
                                  .2, 12.))
        self.pan += event.position() - self.to_widget(before.x(), before.y())
        self.update()
