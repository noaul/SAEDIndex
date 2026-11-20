"""Chinese scientific workbench for diffraction calibration and indexing."""
import copy
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QPixmap, QPainter, QPalette
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from .canvas import PatternCanvas
from .diagnostics import SCORE_LABELS, group_diagnostics
from .rendering import FigureStyle, chemical_text, ensure_fonts, export_figure, qimage_from_array
from .project import SETTINGS
from .session import BASIS_CHOICES, Session, basis_text

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_CIFS = ["Ni_mp-23", "NbNi3_mp-11513", "达尔塔NbNi3_mp-1451", "NbC_mp-910"]
HISTORY_LIMIT = 60
CALIBRATION_STATE = ("px_per_unit", "source", "confident", "reference_points",
                     "reference_value", "reference_unit", "conflict", "proposed_px_per_unit")
VIEWS = (("source", "原图"), ("pattern", "FFT"), ("ifft", "IFFT"), ("compare", "对照"))
SAED_ONLY_MODES = ("center",)
HRTEM_ONLY_MODES = ("roi", "ifft_pick")
SOURCE_VIEW_MODES = ("roi", "scale", "mask")


@lru_cache(maxsize=64)
def icon(name):
    path = ASSETS / "icons" / f"{name}.svg"
    if not path.exists():
        return QIcon()
    root = ET.parse(path).getroot()
    root.set("stroke", "#33403f")
    renderer = QSvgRenderer(ET.tostring(root))
    pixmap = QPixmap(48, 48)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


def button(text, name, slot):
    b = QPushButton(icon(name), text)
    b.clicked.connect(slot)
    return b


def table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().hide()
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    t.setAlternatingRowColors(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def cell(value):
    item = QTableWidgetItem(str(value))
    item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    return item


class ElidedLabel(QLabel):
    """Single-line label that shortens long file names in the middle."""
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideMiddle, self.width())
        painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)

    def minimumSizeHint(self):
        return QSize(20, super().minimumSizeHint().height())


class AnalysisWorker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, session, path=None, parent=None, kind="saed"):
        super().__init__(parent)
        self.session = copy.deepcopy(session, {id(p): p for p in session.phases})
        self.path, self.kind = path, kind

    def run(self):
        try:
            if self.path:
                if self.kind == "hrtem":
                    self.session.open_hrtem(self.path)
                else:
                    self.session.open(self.path)
            self.session.analyse()
            self.result.emit(self.session)
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self, load_defaults=True):
        super().__init__()
        ensure_fonts()
        self.session, self.style = Session(), FigureStyle()
        self.worker = None
        self._updating = False
        self.current_set = 0
        self._reference_image = None
        self._reference_dirty = False
        self._undo, self._redo, self._pending = [], [], None
        self._analysis_before = None
        self._analysis_new_image = True
        self.setWindowTitle("SAEDIndex | 衍射标定")
        self.resize(1480, 960)
        self.setMinimumSize(1000, 700)
        self._build()
        self._style()
        if load_defaults:
            for stem in DEFAULT_CIFS:
                path = Path("D:/cif") / f"{stem}_conventional_standard.cif"
                if path.exists():
                    self.session.add_phase(path)
        self.refresh()

    @property
    def selected_set(self):
        if 0 <= self.current_set < len(self.session.sets):
            return self.session.sets[self.current_set]
        return None

    def _build(self):
        toolbar = QToolBar("文件与标定")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.open_action = toolbar.addAction(icon("folder-open"), "导入图片", self.choose_image)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setToolTip("导入衍射图 (Ctrl+O)")
        self.hrtem_action = toolbar.addAction(icon("microscope"), "导入 HRTEM", self.choose_hrtem)
        self.hrtem_action.setShortcut("Ctrl+Shift+O")
        self.hrtem_action.setToolTip("导入高分辨像，FFT 标定与 IFFT 重构 (Ctrl+Shift+O)")
        self.cif_action = toolbar.addAction(icon("atom"), "导入 CIF", self.choose_cifs)
        toolbar.addSeparator()
        self.run_action = toolbar.addAction(icon("wand-sparkles"), "自动识别", self.analyse)
        self.run_action.setToolTip("重新检测斑点、中心及晶格，并搜索 CIF 候选")
        toolbar.addAction(icon("maximize"), "适应窗口", lambda: self.canvas.fit())
        self.markers_action = toolbar.addAction(icon("eye"), "隐藏标记")
        self.markers_action.setCheckable(True)
        self.markers_action.setChecked(True)
        self.markers_action.setShortcut("H")
        self.markers_action.toggled.connect(self.set_markers_visible)
        self.markers_action.setToolTip("隐藏斑点圈、中心、端点及模拟框 (H)")
        toolbar.addSeparator()
        self.undo_action = toolbar.addAction(icon("undo-2"), "撤销", self.undo)
        self.undo_action.setShortcuts(["Ctrl+Z"])
        self.redo_action = toolbar.addAction(icon("redo-2"), "重做", self.redo)
        self.redo_action.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        toolbar.addSeparator()
        self.csv_action = toolbar.addAction(icon("file-spreadsheet"), "导出数据", self.export_csv)
        toolbar.addSeparator()
        self.save_action = toolbar.addAction(icon("save"), "保存项目", self.save_project)
        toolbar.addAction(icon("file-text"), "打开项目", self.load_project)

        modes = QToolBar("图像编辑")
        modes.setIconSize(QSize(20, 20))
        modes.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, modes)
        group = QActionGroup(self)
        self.mode_actions = {}
        for name, title, glyph in [
            ("pan", "平移", "hand"), ("inspect", "斑点数据", "mouse-pointer-2"),
            ("center", "设定透射中心", "crosshair"), ("add", "补充斑点", "circle-plus"),
            ("delete", "删除斑点", "circle-minus"), ("move", "移动斑点", "move"),
            ("pair", "双斑点建组", "pentagon"), ("delete_group", "点击删除组", "trash-2"),
            ("labels", "移动标注", "type"),
            ("scale", "定标尺端点", "ruler"), ("mask", "出图覆盖区域", "scissors"),
            ("roi", "框选 FFT 区域", "scan"), ("ifft_pick", "选取 IFFT 斑点", "target"),
        ]:
            a = modes.addAction(icon(glyph), title)
            a.setCheckable(True)
            a.setToolTip(title)
            group.addAction(a)
            a.triggered.connect(lambda checked=False, mode=name: self.activate_mode(mode))
            self.mode_actions[name] = a
        self.mode_actions["pan"].setChecked(True)
        self.splitter = QSplitter()
        self.setCentralWidget(self.splitter)
        self.sidebar = self._sidebar()
        self.splitter.addWidget(self.sidebar)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        header = QWidget()
        header.setObjectName("DisplayBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 7, 12, 7)
        self.filename = ElidedLabel("未导入图片")
        self._filename_text = "未导入图片"
        self.filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        header_layout.addWidget(self.filename, 1)
        self.view_bar = QWidget()
        self.view_bar.setObjectName("ViewBar")
        view_layout = QHBoxLayout(self.view_bar)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_buttons = {}
        for k, (name, title) in enumerate(VIEWS):
            b = QToolButton()
            b.setText(title)
            b.setCheckable(True)
            b.setObjectName("Segment")
            b.setMinimumWidth(54)
            b.clicked.connect(lambda checked=False, view=name: self.set_view(view))
            self.view_group.addButton(b)
            self.view_buttons[name] = b
            view_layout.addWidget(b)
        self.view_buttons["pattern"].setChecked(True)
        self.view_bar.hide()
        header_layout.addWidget(self.view_bar)
        self.zoom_label = QLabel("")
        self.zoom_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        header_layout.addWidget(self.zoom_label)
        self.spots_action = QAction(icon("circle-minus"), "斑点圈", self)
        self.spots_action.setCheckable(True)
        self.spots_action.setChecked(self.style.show_spots)
        self.spots_action.setToolTip("独立显示 / 隐藏检测斑点圈，保留 CIF 模拟与标注")
        self.spots_action.toggled.connect(self.show_spots.setChecked)
        self.spots_button = QToolButton()
        self.spots_button.setDefaultAction(self.spots_action)
        self.spots_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header_layout.addWidget(self.spots_button)
        center_layout.addWidget(header)
        self.canvas = PatternCanvas()
        self.canvas.message.connect(self.say)
        self.canvas.changed.connect(self.geometry_changed)
        self.canvas.edit_started.connect(self._begin_edit)
        self.canvas.labels_moved.connect(lambda: self._commit_edit("移动标注"))
        self.canvas.pair_selected.connect(self.add_pair_set)
        self.canvas.reference_selected.connect(self.stage_scale)
        self.canvas.group_delete_requested.connect(self.delete_group_at)
        self.canvas.mask_selected.connect(self.set_mask)
        self.canvas.inspected.connect(self.inspect_spot)
        self.canvas.roi_selected.connect(self.apply_roi)
        self.canvas.ifft_toggled.connect(self.toggle_ifft_spot)
        center_layout.addWidget(self.canvas, 1)
        self.spot_info = QLabel(" ")
        self.spot_info.setContentsMargins(12, 6, 12, 6)
        self.spot_info.setWordWrap(True)
        center_layout.addWidget(self.spot_info)
        self.splitter.addWidget(center)
        self.right_panel = self._right_panel()
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([265, 660, 500])
        self.splitter.setStretchFactor(1, 1)
        self.statusBar().showMessage("就绪")

    def _section(self, layout, text):
        title = QLabel(text)
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

    def _settings_group(self, parent, title):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 18, 8, 10)
        layout.setSpacing(8)
        parent.addWidget(box)
        return layout

    def _spin(self, low, high, value, decimals=2, suffix=""):
        b = QDoubleSpinBox()
        b.setRange(low, high)
        b.setDecimals(decimals)
        b.setValue(value)
        b.setSuffix(suffix)
        b.setKeyboardTracking(False)
        return b

    def _sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(225)
        scroll.setMaximumWidth(345)
        panel = QWidget()
        panel.setObjectName("Settings")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 10, 8, 10)
        self.sidebar_tabs = QTabWidget()
        outer.addWidget(self.sidebar_tabs)
        pages = []
        self.setting_scrolls = []
        for title in ("定标", "识别", "显示"):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 10, 4, 4)
            page_layout.setSpacing(12)
            page_scroll = QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setWidget(page)
            page_scroll.setMinimumWidth(0)
            self.setting_scrolls.append(page_scroll)
            self.sidebar_tabs.addTab(page_scroll, title)
            pages.append(page_layout)
        calibration_page, detection_page, display_page = pages
        layout = self._settings_group(calibration_page, "标尺识别证据")
        self.cal_source = QLabel("未标定")
        self.cal_source.setWordWrap(True)
        self.cal_source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.cal_source)
        self.cal_evidence = QLabel("")
        self.cal_evidence.setWordWrap(True)
        self.cal_evidence.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.cal_evidence)
        self.cal_thumbnail = QLabel()
        self.cal_thumbnail.setFixedHeight(70)
        self.cal_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cal_thumbnail.hide()
        layout.addWidget(self.cal_thumbnail)
        self.cal_notes = QLabel("")
        self.cal_notes.setWordWrap(True)
        self.cal_notes.setObjectName("Warning")
        layout.addWidget(self.cal_notes)
        layout = self._settings_group(calibration_page, "科学定标")
        ref = QFormLayout()
        ref.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.reference_value = self._spin(.0001, 100000, 5, 4)
        self.reference_value.valueChanged.connect(self.reference_changed)
        self.reference_unit = QComboBox()
        self.reference_unit.addItems(["1/nm", "1/\u00c5"])
        self.reference_unit.currentIndexChanged.connect(self.reference_changed)
        ref.addRow("原标尺数值", self.reference_value)
        self.reference_unit_label = QLabel("倒易单位")
        ref.addRow(self.reference_unit_label, self.reference_unit)
        self.reference_coordinates = []
        for title in ("起点 x / y", "终点 x / y"):
            row = QHBoxLayout()
            for _ in range(2):
                spin = self._spin(0, 100000, 0, 2)
                spin.setMinimumWidth(0)
                spin.setToolTip(f"{title} 像素坐标")
                spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                spin.valueChanged.connect(self.reference_changed)
                self.reference_coordinates.append(spin)
                row.addWidget(spin)
            ref.addRow(QLabel(title))
            ref.addRow(row)
        self.reference_pixels = self._spin(0, 100000, 0, 3, " px")
        self.reference_pixels.valueChanged.connect(self.reference_length_changed)
        ref.addRow("有效长度", self.reference_pixels)
        layout.addLayout(ref)
        self.reference_result = QLabel("")
        self.reference_result.setWordWrap(True)
        layout.addWidget(self.reference_result)
        layout.addWidget(button("选取 / 调整端点", "ruler", lambda: self.activate_mode("scale")))
        self.reference_apply = button("应用标尺定标", "check", self.apply_reference)
        row = QHBoxLayout()
        row.addWidget(self.reference_apply, 1)
        self.reference_restore = button("", "rotate-ccw", self.restore_reference)
        self.reference_restore.setToolTip("恢复已用定标，放弃尚未应用的端点修改")
        row.addWidget(self.reference_restore)
        layout.addLayout(row)
        self.show_calibration = QCheckBox("显示定标端点")
        self.show_calibration.setChecked(True)
        self.show_calibration.toggled.connect(self.display_changed)
        layout.addWidget(self.show_calibration)
        layout = self._settings_group(calibration_page, "直接输入比例")
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.cal_value = self._spin(0, 1000000, 0, 5)
        self.cal_value.setSpecialValueText("待输入")
        self.cal_value.setMinimumWidth(72)
        self.cal_value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.cal_value.setToolTip("每 1/nm 对应的像素数，不是 nm/px")
        self.cal_value_label = QLabel("px / (1/nm)")
        form.addRow(self.cal_value_label, self.cal_value)
        layout.addLayout(form)
        layout.addWidget(button("直接应用像素比例", "check", self.apply_calibration))
        calibration_page.addStretch()
        self.hrtem_box = QGroupBox("FFT 区域与 IFFT")
        detection_page.addWidget(self.hrtem_box)
        layout = QVBoxLayout(self.hrtem_box)
        layout.setContentsMargins(8, 18, 8, 10)
        layout.setSpacing(8)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.roi_fields = []
        for title, tip in (("ROI x", "ROI 左上角 x / px"), ("ROI y", "ROI 左上角 y / px"),
                           ("边长", "正方形 ROI 边长 / px")):
            spin = QSpinBox()
            spin.setRange(0, 100000)
            spin.setSuffix(" px")
            spin.setToolTip(tip)
            spin.setMinimumWidth(72)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setKeyboardTracking(False)
            self.roi_fields.append(spin)
            form.addRow(title, spin)
        layout.addLayout(form)
        row = QHBoxLayout()
        roi_pick = button("框选", "scan", lambda: self.activate_mode("roi"))
        roi_pick.setToolTip("在原图上拖出正方形 FFT 区域")
        row.addWidget(roi_pick)
        self.roi_apply = button("应用 ROI", "check", self.apply_roi_fields)
        self.roi_apply.setToolTip("重新计算 FFT、检测斑点并索引；可撤销")
        row.addWidget(self.roi_apply)
        layout.addLayout(row)
        self.fft_info = QLabel("")
        self.fft_info.setWordWrap(True)
        layout.addWidget(self.fft_info)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.fft_zoom = QComboBox()
        for text, value in (("自动", 0), ("1×", 1), ("2×", 2), ("4×", 4), ("8×", 8)):
            self.fft_zoom.addItem(text, value)
        self.fft_zoom.setToolTip("在零频附近加密采样（等效补零），不改变倒易标定")
        self.fft_zoom.currentIndexChanged.connect(self.fft_zoom_changed)
        form.addRow("FFT 倍率", self.fft_zoom)
        self.ifft_radius = self._spin(.5, 200, 4, 1, " px")
        self.ifft_radius.setToolTip("掩膜半径，单位为未放大的 FFT 像素")
        self.ifft_soft = self._spin(0, 100, 2, 1, " px")
        self.ifft_soft.setToolTip("掩膜余弦软边宽度，减小振铃")
        for spin in (self.ifft_radius, self.ifft_soft):
            spin.setMinimumWidth(72)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.valueChanged.connect(self.ifft_mask_changed)
        form.addRow("掩膜半径", self.ifft_radius)
        form.addRow("软边", self.ifft_soft)
        layout.addLayout(form)
        row = QHBoxLayout()
        pick = button("选点", "target", lambda: self.activate_mode("ifft_pick"))
        pick.setToolTip("在 FFT 上点击斑点，加入或移出 IFFT（自动包含 Friedel 对称点）")
        row.addWidget(pick)
        group_pick = button("", "layers", self.ifft_use_group)
        group_pick.setToolTip("仅用当前斑点组的斑点重构")
        row.addWidget(group_pick)
        all_pick = button("", "grid-2x2", self.ifft_use_all)
        all_pick.setToolTip("使用全部检测斑点")
        row.addWidget(all_pick)
        for b in (roi_pick, self.roi_apply, pick):
            b.setMinimumWidth(0)
            b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addLayout(row)
        self.ifft_button = button("计算 IFFT", "play", lambda: self.set_view("ifft", recompute=True))
        self.ifft_button.setObjectName("Primary")
        layout.addWidget(self.ifft_button)
        self.ifft_info = QLabel("")
        self.ifft_info.setWordWrap(True)
        layout.addWidget(self.ifft_info)
        caveat = QLabel("FFT 周期可能来自莫尔条纹或半周期衬度；IFFT 为所选频率的滤波结果，不含新信息。")
        caveat.setWordWrap(True)
        caveat.setObjectName("Warning")
        layout.addWidget(caveat)
        self.hrtem_box.hide()
        layout = self._settings_group(detection_page, "亮点检测")
        form = QFormLayout()
        self.detection_form = form
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.sensitivity = self._spin(2, 30, 6, 1)
        self.sensitivity.setToolTip("相对于背景噪声的检测阈值；调高可减少弱点")
        self.min_distance = QSpinBox()
        self.min_distance.setRange(3, 100)
        self.min_distance.setValue(6)
        self.min_distance.setToolTip("相邻峰的最小间隔，不能替代亮点直径筛选")
        self.min_diameter = self._spin(0, 200, 0, 1, " px")
        self.max_diameter = self._spin(0, 500, 0, 1, " px")
        for field in (self.min_diameter, self.max_diameter):
            field.setSpecialValueText("不限")
            field.setToolTip("原图像素中的半峰高连通面积等效直径；0 表示不限，不是晶面间距")
            field.valueChanged.connect(self.detection_parameters_changed)
        self.smoothing = self._spin(.3, 5, 1.2, 1, " px")
        self.smoothing.setToolTip("检测前的高斯平滑 sigma；不会修改原图或显示亮度")
        for field in (self.sensitivity, self.min_diameter, self.max_diameter, self.smoothing):
            field.setMinimumWidth(72)
            field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for field in (self.sensitivity, self.min_distance, self.smoothing):
            field.valueChanged.connect(self.detection_parameters_changed)
        form.addRow("亮点阈值", self.sensitivity)
        form.addRow("最小直径", self.min_diameter)
        form.addRow("最大直径", self.max_diameter)
        form.addRow("峰间隔 / px", self.min_distance)
        form.addRow("平滑尺度", self.smoothing)
        layout.addLayout(form)
        self.detection_status = QLabel("")
        self.detection_status.setWordWrap(True)
        layout.addWidget(self.detection_status)
        self.redetect_button = button("重新识别斑点", "wand-sparkles", self.analyse)
        self.redetect_button.setObjectName("Primary")
        self.redetect_button.setToolTip("按当前参数重新检测、分组与索引；可撤销恢复原标定")
        layout.addWidget(self.redetect_button)
        layout.addWidget(button("恢复检测默认值", "rotate-ccw", self.reset_detection))
        layout = self._settings_group(detection_page, "候选物相")
        self.phase_list = QListWidget()
        self.phase_list.setMaximumHeight(145)
        self.phase_list.setMinimumHeight(92)
        self.phase_list.itemChanged.connect(self.phase_changed)
        layout.addWidget(self.phase_list)
        row = QHBoxLayout()
        row.addWidget(button("CIF", "file-plus", self.choose_cifs))
        row.addWidget(button("移除", "trash-2", self.remove_phase))
        layout.addLayout(row)

        layout = self._settings_group(detection_page, "晶格与匹配")
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.max_sets = QSpinBox()
        self.max_sets.setRange(1, 6)
        self.max_sets.setValue(3)
        self.ratio_tol = self._spin(.1, 20, 3, 1, " %")
        self.scale_tol = self._spin(.1, 30, 6, 1, " %")
        self.angle_tol = self._spin(.1, 15, 3, 1, "°")
        form.addRow("斑点组数上限", self.max_sets)
        form.addRow("长度比容差", self.ratio_tol)
        form.addRow("整体比例容差", self.scale_tol)
        form.addRow("角度容差", self.angle_tol)
        layout.addLayout(form)
        self.forbidden = QCheckBox("包含平移消光候选")
        self.forbidden.setToolTip("通常保持关闭。开启可能引入不符合晶格平移对称性的假匹配。")
        layout.addWidget(self.forbidden)
        detection_page.addStretch()
        layout = self._settings_group(display_page, "图像显示")
        self.brightness = self._slider(layout, "亮度", -100, 100, 0)
        self.contrast = self._slider(layout, "对比度", 10, 400, 100)
        self.gamma = self._slider(layout, "Gamma", 20, 300, 100)
        self.log_display = QCheckBox("对数显示")
        self.log_display.toggled.connect(self.display_changed)
        layout.addWidget(self.log_display)
        layout.addWidget(button("重置显示", "rotate-ccw", self.reset_display))
        layout = self._settings_group(display_page, "标记图层")
        self.show_spots = QCheckBox("检测斑点圈")
        self.show_spots.setChecked(True)
        self.show_spots.toggled.connect(self.display_changed)
        layout.addWidget(self.show_spots)
        self.show_annotations = QCheckBox("晶面与晶带轴标注")
        self.show_annotations.setChecked(True)
        self.show_annotations.toggled.connect(self.display_changed)
        layout.addWidget(self.show_annotations)
        self.center_box = QCheckBox("透射中心")
        self.center_box.setChecked(True)
        self.center_box.toggled.connect(self.display_changed)
        layout.addWidget(self.center_box)
        display_page.addStretch()
        scroll.setWidget(panel)
        return scroll

    def detection_parameters_changed(self, *args):
        if self._updating or not hasattr(self, "detection_status"):
            return
        invalid = self.max_diameter.value() > 0 and self.min_diameter.value() > self.max_diameter.value()
        pending = any(getattr(self, key).value() != getattr(self.session, key)
                      for key in ("sensitivity", "min_distance", "min_diameter", "max_diameter", "smoothing"))
        self.detection_status.setText("最大直径不能小于最小直径" if invalid else
                                      "参数已修改，尚未重新识别" if pending else
                                      f"已检测 {len(self.session.spots)} 个斑点" if self.session.image else "尚未导入图片")
        self.redetect_button.setEnabled(not invalid and self.session.image is not None and not self.worker)

    def reset_detection(self):
        for widget, value in ((self.sensitivity, 6), (self.min_distance, 6),
                              (self.min_diameter, 0), (self.max_diameter, 0), (self.smoothing, 1.2)):
            widget.setValue(value)

    def _slider(self, layout, title, low, high, value):
        row = QHBoxLayout()
        label = QLabel(title)
        label.setMinimumWidth(52)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(self.display_changed)
        row.addWidget(label)
        row.addWidget(slider, 1)
        layout.addLayout(row)
        return slider

    def _right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(650)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(8)
        main_layout = layout
        layout = self._settings_group(main_layout, "当前斑点组")
        row = QHBoxLayout()
        self.set_combo = QComboBox()
        self.set_combo.currentIndexChanged.connect(self.change_set)
        row.addWidget(self.set_combo, 1)
        self.visible_box = QCheckBox("显示")
        self.visible_box.setChecked(True)
        self.visible_box.toggled.connect(self.set_visibility)
        row.addWidget(self.visible_box)
        self.add_group_button = button("", "circle-plus", lambda: self.activate_mode("pair"))
        self.add_group_button.setToolTip("添加组：依次选择两个不共线斑点")
        row.addWidget(self.add_group_button)
        self.delete_group_button = button("", "trash-2", self.delete_set)
        self.delete_group_button.setToolTip("删除当前斑点组，保留原始斑点")
        row.addWidget(self.delete_group_button)
        layout.addLayout(row)
        self.set_summary = QLabel("")
        self.set_summary.setWordWrap(True)
        layout.addWidget(self.set_summary)
        self.analysis_status = QLabel("")
        self.analysis_status.setWordWrap(True)
        layout.addWidget(self.analysis_status)
        layout = main_layout
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(200)
        self.tabs.setMaximumHeight(260)
        layout.addWidget(self.tabs, 1)
        self.candidate_table = table(["物相", "晶带轴", "Δg1 %", "Δg2 %", "Δθ °", "评分"])
        self.candidate_table.setToolTip("误差为 |g|实测/|g|CIF−1；评分越低越优，不代表概率。")
        self.candidate_table.itemSelectionChanged.connect(self.choose_candidate)
        self.tabs.addTab(self.candidate_table, "CIF 候选")
        self.vector_table = table(["矢量", "hkl", "d实测 nm", "dCIF nm", "Δd %", "θ实测°", "θCIF°", "来源"])
        self.tabs.addTab(self.vector_table, "矢量对比")
        self.spots_table = table(["斑点", "hkl", "d实测 nm", "Δd %", "亮度", "类型"])
        self.spots_table.itemSelectionChanged.connect(self.select_spot_row)
        self.tabs.addTab(self.spots_table, "全部斑点")
        diagnosis_page = QWidget()
        diagnosis_layout = QVBoxLayout(diagnosis_page)
        diagnosis_layout.setContentsMargins(6, 6, 6, 6)
        self.diagnostics_summary = QLabel("")
        self.diagnostics_summary.setWordWrap(True)
        self.diagnostics_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        diagnosis_layout.addWidget(self.diagnostics_summary)
        self.score_details = QLabel("")
        self.score_details.setWordWrap(True)
        self.score_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        diagnosis_layout.addWidget(self.score_details)
        self.grouping_notes = QLabel("")
        self.grouping_notes.setWordWrap(True)
        diagnosis_layout.addWidget(self.grouping_notes)
        self.diagnostics_table = table(["斑点", "归属", "晶格残差 px", "Δd %"])
        self.diagnostics_table.itemSelectionChanged.connect(self.select_diagnostic_row)
        diagnosis_layout.addWidget(self.diagnostics_table, 1)
        self.tabs.addTab(diagnosis_page, "诊断")
        self.closure = QLabel("")
        self.closure.setWordWrap(True)
        layout.addWidget(self.closure)
        self.match_warning = QLabel("")
        self.match_warning.setWordWrap(True)
        self.match_warning.setObjectName("Warning")
        layout.addWidget(self.match_warning)
        self.confirm_button = button("确认当前候选", "circle-check", self.confirm_set)
        self.confirm_button.setObjectName("Primary")
        layout.addWidget(self.confirm_button)

        preview_row = QHBoxLayout()
        self.preview_box = QCheckBox("论文出图预览")
        self.preview_box.toggled.connect(self.display_changed)
        preview_row.addWidget(self.preview_box)
        self.export_action = QAction(icon("image-down"), "导出标定图", self)
        self.export_action.triggered.connect(self.export_image)
        self.preview_export_button = QToolButton()
        self.preview_export_button.setDefaultAction(self.export_action)
        self.preview_export_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        preview_row.addWidget(self.preview_export_button)
        preview_row.addStretch()
        layout.addLayout(preview_row)
        self.figure_tabs = QTabWidget()
        layout.addWidget(self.figure_tabs)
        simulation_page = QWidget()
        sim_layout = QVBoxLayout(simulation_page)
        sim_layout.setContentsMargins(6, 8, 6, 6)
        sim_layout = self._settings_group(sim_layout, "CIF 叠加与尺度")
        self.figure_tabs.addTab(simulation_page, "CIF 模拟")
        row = QHBoxLayout()
        self.simulation_box = QCheckBox("CIF 位置叠加")
        self.simulation_box.toggled.connect(self.simulation_changed)
        self.free_scale_box = QCheckBox("允许比例拟合")
        self.free_scale_box.setToolTip("只改变模拟，不改图像标定。关闭时严格使用标尺标定值。")
        self.free_scale_box.toggled.connect(self.simulation_changed)
        row.addWidget(self.simulation_box)
        row.addWidget(self.free_scale_box)
        sim_layout.addLayout(row)
        self.sim_info = QLabel("")
        self.sim_info.setWordWrap(True)
        sim_layout.addWidget(self.sim_info)
        self.residual_box = QCheckBox("匹配状态与残差")
        self.residual_box.setToolTip("青色为匹配点，橙色为偏差点，灰色为未检出点；连线使用固定科学定标")
        self.residual_box.toggled.connect(self.residuals_changed)
        sim_layout.addWidget(self.residual_box)
        self.cif_scale_info = QLabel("")
        self.cif_scale_info.setWordWrap(True)
        sim_layout.addWidget(self.cif_scale_info)
        self.cif_scale_button = button("应用 CIF 辅助比例", "ruler", self.apply_cif_scale)
        self.cif_scale_button.setToolTip("将所选 CIF 作为比例参考，重新索引并清除确认；支持撤销")
        sim_layout.addWidget(self.cif_scale_button)
        sim_form = QFormLayout()
        self.sim_size = self._spin(3, 20, 7, 1, " px")
        self.sim_size.valueChanged.connect(self.display_changed)
        self.sim_opacity = self._spin(10, 100, 95, 0, " %")
        self.sim_opacity.valueChanged.connect(self.display_changed)
        sim_form.addRow("模拟框大小", self.sim_size)
        sim_form.addRow("不透明度", self.sim_opacity)
        sim_layout.addLayout(sim_form)
        self.dynamic_box = QCheckBox("显示动力学可能反射")
        self.dynamic_box.toggled.connect(self.display_changed)
        sim_layout.addWidget(self.dynamic_box)
        sim_layout.addStretch()
        annotation_page = QWidget()
        ann_layout = QVBoxLayout(annotation_page)
        ann_layout.setContentsMargins(6, 8, 6, 6)
        ann_layout = self._settings_group(ann_layout, "标注内容与样式")
        self.figure_tabs.addTab(annotation_page, "标注样式")
        form = QFormLayout()
        self.equivalent_combo = QComboBox()
        self.equivalent_combo.currentIndexChanged.connect(self.equivalent_changed)
        form.addRow("等价标法", self.equivalent_combo)
        self.basis_combo = QComboBox()
        self.basis_combo.addItems([basis_text(p) for p in BASIS_CHOICES])
        self.basis_combo.currentIndexChanged.connect(self.basis_changed)
        form.addRow("平行四边形", self.basis_combo)
        self.title_edit = QLineEdit()
        self.title_edit.setToolTip("留空时使用当前候选的物相名称")
        self.title_edit.editingFinished.connect(self.title_changed)
        form.addRow("物相后缀", self.title_edit)
        self.chemical_subscripts = QCheckBox("化学式数字下标")
        self.chemical_subscripts.setChecked(True)
        self.chemical_subscripts.setToolTip("将 NbNi3 等化学式的数字显示为下标；不改变 CIF 或晶面指数")
        self.chemical_subscripts.toggled.connect(self.display_changed)
        self.phase_subscript = QCheckBox("物相名称作下标")
        self.phase_subscript.setChecked(True)
        self.phase_subscript.toggled.connect(self.display_changed)
        form.addRow("物相排版", self.phase_subscript)
        form.addRow("", self.chemical_subscripts)
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 200)
        self.font_size.setValue(38)
        self.font_size.valueChanged.connect(self.display_changed)
        form.addRow("字高 / px", self.font_size)
        self.font_bold = QCheckBox("加粗")
        self.font_bold.toggled.connect(self.display_changed)
        form.addRow("字体", self.font_bold)
        self.line_width = self._spin(.25, 20, 2.5, 2, " px")
        self.line_width.valueChanged.connect(self.display_changed)
        form.addRow("标定线宽", self.line_width)
        ann_layout.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(button("标注颜色", "contrast", self.choose_color))
        row.addWidget(button("指定 hkl", "pen-line", self.manual_indices))
        ann_layout.addLayout(row)
        ann_layout.addStretch()
        bar_page = QWidget()
        bar_layout = QVBoxLayout(bar_page)
        bar_layout.setContentsMargins(6, 8, 6, 6)
        bar_layout = self._settings_group(bar_layout, "论文出图标尺")
        self.figure_tabs.addTab(bar_page, "出图标尺")
        form = QFormLayout()
        self.bar_value = self._spin(.001, 10000, 5, 3)
        self.bar_value.valueChanged.connect(self.display_changed)
        form.addRow("新标尺 / (1/nm)", self.bar_value)
        self.real_bar_value = self._spin(.001, 100000, 5, 3)
        self.real_bar_value.valueChanged.connect(self.display_changed)
        self.real_bar_label = QLabel("实空间标尺 / nm")
        form.addRow(self.real_bar_label, self.real_bar_value)
        self.bar_form = form
        self.bar_font_size = QSpinBox()
        self.bar_font_size.setRange(8, 200)
        self.bar_font_size.setValue(56)
        self.bar_font_size.valueChanged.connect(self.display_changed)
        form.addRow("标尺字高 / px", self.bar_font_size)
        self.bar_thickness = QSpinBox()
        self.bar_thickness.setRange(1, 40)
        self.bar_thickness.setValue(12)
        self.bar_thickness.valueChanged.connect(self.display_changed)
        form.addRow("标尺粗细 / px", self.bar_thickness)
        bar_layout.addLayout(form)
        self.replace_bar = QCheckBox("覆盖并替换原标尺")
        self.replace_bar.setChecked(True)
        self.replace_bar.toggled.connect(self.display_changed)
        bar_layout.addWidget(self.replace_bar)
        bar_layout.addWidget(button("框选原标尺覆盖区域", "scissors",
                                   lambda: self.activate_mode("mask")))
        bar_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(650)
        scroll.setWidget(panel)
        return scroll

    def _style(self):
        palette = self.palette()
        for role, color in ((QPalette.ColorRole.Window, "#ffffff"),
                            (QPalette.ColorRole.Base, "#ffffff"),
                            (QPalette.ColorRole.AlternateBase, "#f5f7f7"),
                            (QPalette.ColorRole.Text, "#1f2627"),
                            (QPalette.ColorRole.WindowText, "#1f2627"),
                            (QPalette.ColorRole.Button, "#f3f5f5"),
                            (QPalette.ColorRole.ButtonText, "#1f2627"),
                            (QPalette.ColorRole.Highlight, "#1f7a6e"),
                            (QPalette.ColorRole.HighlightedText, "#ffffff")):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        self.setStyleSheet("""
        QWidget { font-family: "Microsoft YaHei UI"; font-size: 12px; color:#1f2627; background:#ffffff; }
        QLabel { background:transparent; }
        QLabel#SectionTitle { font-size:14px; font-weight:600; padding-top:4px; }
        QLabel#Warning { color:#b45a09; font-size:12px; }
        QGroupBox { border:1px solid #dbe1e1; border-radius:6px; margin-top:9px; background:#f8fafa; }
        QGroupBox::title { subcontrol-origin:margin; left:9px; padding:0 5px; color:#1b2b2a; font-weight:600; }
        QWidget#Settings, QWidget#DisplayBar { background:#ffffff; }
        QToolBar { background:#f6f8f8; padding:5px; spacing:5px; border:0; border-bottom:1px solid #e1e6e6; }
        QToolButton { padding:6px; border:1px solid transparent; border-radius:4px; background:transparent; }
        QToolButton:hover { background:#e7eeed; }
        QToolButton:checked { background:#d4ebe7; border-color:#5baba0; }
        QPushButton { padding:6px 8px; border:1px solid #c9d1d1; border-radius:4px; background:#ffffff; }
        QPushButton:hover { background:#eef5f4; border-color:#6d9690; }
        QPushButton:pressed { background:#d4ebe7; }
        QPushButton:disabled { color:#a3abac; background:#f4f6f6; border-color:#e0e5e5; }
        QPushButton#Primary { background:#1f7a6e; border-color:#1a6a5f; color:white; font-weight:600; }
        QPushButton#Primary:hover { background:#268a7d; }
        QPushButton#Primary:disabled { background:#c8dcd8; color:#f4f8f7; border-color:#bcd2ce; }
        QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit { min-height:24px; padding:1px 4px;
          background:#ffffff; border:1px solid #c9d1d1; border-radius:4px; selection-background-color:#1f7a6e; }
        QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus,QLineEdit:focus,QPushButton:focus { border-color:#2a9585; }
        QSpinBox:disabled,QDoubleSpinBox:disabled,QComboBox:disabled,QLineEdit:disabled { color:#a3abac; background:#f4f6f6; }
        QCheckBox { spacing:6px; background:transparent; }
        QTableWidget { border:0; background:#ffffff; alternate-background-color:#f5f7f7;
          selection-background-color:#d4ebe7; selection-color:#13302c; gridline-color:#e3e8e8; }
        QTableWidget::item:selected { background:#d4ebe7; color:#13302c; }
        QHeaderView::section { background:#eef2f2; padding:6px 4px; border:0; color:#3c4849; }
        QTabWidget::pane { border:0; }
        QTabBar::tab { background:#f3f5f5; padding:8px 9px; color:#55605f; border-bottom:2px solid transparent; }
        QTabBar::tab:selected { background:#ffffff; color:#1a6a5f; border-bottom-color:#1f7a6e; }
        QTabBar::tab:hover { background:#e7eeed; }
        QSplitter::handle { background:#dde3e3; width:1px; }
        QScrollArea { border:0; }
        QListWidget { border:1px solid #dbe1e1; background:#ffffff; border-radius:4px;
          selection-background-color:#d4ebe7; selection-color:#13302c; }
        QStatusBar { background:#f6f8f8; border-top:1px solid #e1e6e6; color:#4d5859; }
        QSlider::groove:horizontal { height:4px; background:#d3dada; }
        QSlider::handle:horizontal { background:#1f7a6e; width:12px; margin:-5px 0; border-radius:5px; }
        QToolTip { color:#1f2627; background:#ffffff; border:1px solid #c9d1d1; padding:5px; }
        QToolButton#Segment { border:1px solid #c9d1d1; border-radius:0; padding:4px 10px;
          background:#ffffff; color:#3c4849; }
        QToolButton#Segment:checked { background:#1f7a6e; border-color:#1f7a6e; color:#ffffff; }
        QToolButton#Segment:hover { background:#e7eeed; }
        """)

    def say(self, text):
        self.statusBar().showMessage(text)

    def activate_mode(self, mode):
        if self.session.image is None or self.worker:
            self.say("尚无可编辑图片。")
            return
        hrtem = self.session.mode == "hrtem"
        if (hrtem and mode in SAED_ONLY_MODES) or (not hrtem and mode in HRTEM_ONLY_MODES):
            self.say("该工具不适用于当前图像类型。")
            return
        if hrtem:
            if mode in SOURCE_VIEW_MODES:
                self.set_view("source")
            elif mode not in ("pan",):
                self.set_view("pattern")
        if mode in ("center", "add", "delete", "move", "pair", "scale", "delete_group"):
            self.preview_box.setChecked(False)
        if mode in ("mask", ):
            self.preview_box.setChecked(True)
        if mode in ("roi", "ifft_pick"):
            self.preview_box.setChecked(False)
        if mode in ("inspect", "add", "delete", "move", "pair", "delete_group", "ifft_pick"):
            self.show_spots.setChecked(True)
        if mode == "scale":
            self.show_calibration.setChecked(True)
        if mode in ("inspect", "center", "add", "delete", "move", "pair", "delete_group", "scale",
                    "roi", "ifft_pick"):
            self.markers_action.setChecked(True)
        self.mode_actions[mode].setChecked(True)
        self.canvas.set_mode(mode)
        messages = {
            "pair": "新增组：选择两个相对透射中心不共线的斑点。",
            "delete_group": "删除组：点击该组斑点或平行四边形；重合时删除当前组。",
            "scale": ("实空间标尺：在原图上选择两端，或拖动已有端点；修改尚未应用。" if hrtem else
                      "标尺定标：选择两端，或拖动已有端点；修改尚未应用。"),
            "labels": "标注位置编辑。",
            "mask": "论文出图覆盖区域编辑。",
            "roi": "FFT 区域：在原图上拖出正方形。",
            "ifft_pick": "IFFT 选点：点击 FFT 斑点加入或移出。",
        }
        self.say(messages.get(mode, self.mode_actions[mode].text()))

    # ------------------------------------------------------------- HRTEM
    def choose_hrtem(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入 HRTEM 高分辨像", "",
            "高分辨像 (*.dm3 *.dm4 *.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if path:
            self.open_image(path, kind="hrtem")

    def set_view(self, view, recompute=False):
        if self.session.mode != "hrtem" or self.session.image is None:
            return
        if view in ("ifft", "compare") and not self.session.ifft_peaks():
            self.say("没有可用于 IFFT 的斑点，请先检测或选点。")
            view = "pattern"
        if recompute and view in ("ifft", "compare"):
            self.session.hrtem.ifft = None
        if view != "source" and self.canvas.mode in SOURCE_VIEW_MODES:
            self.mode_actions["pan"].setChecked(True)
            self.canvas.set_mode("pan")
        if view == "source" and self.canvas.mode not in SOURCE_VIEW_MODES + ("pan",):
            self.mode_actions["pan"].setChecked(True)
            self.canvas.set_mode("pan")
        self.view_buttons[view].setChecked(True)
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.canvas.set_view(view)
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_hrtem()

    def apply_roi(self, roi):
        if self.session.mode != "hrtem" or self.worker:
            return
        if self._reference_dirty:
            self.say("实空间标尺修改尚未应用，请先应用或恢复。")
            return
        before = self._snapshot()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._read_settings()
            self.session.set_roi(roi)
        except ValueError as error:
            self.say(str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._push_history("修改 FFT 区域", before)
        self.current_set = 0
        self.canvas.selected = None
        self.refresh()
        self.say(f"FFT 区域 {roi[2]} px：检测到 {len(self.session.spots)} 个斑点、"
                 f"{len(self.session.sets)} 组晶格。")

    def apply_roi_fields(self):
        self.apply_roi([f.value() for f in self.roi_fields])

    def fft_zoom_changed(self, *args):
        if self._updating or self.session.mode != "hrtem" or self.worker:
            return
        before = self._snapshot()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._read_settings()
            self.session.set_fft_zoom(self.fft_zoom.currentData())
        finally:
            QApplication.restoreOverrideCursor()
        self._push_history("修改 FFT 倍率", before)
        self.current_set = 0
        self.canvas.selected = None
        self.refresh(reset=True)
        self.say(f"FFT 倍率 {self.session.hrtem.fft_zoom}×：检测到 {len(self.session.spots)} 个斑点。")

    def toggle_ifft_spot(self, index):
        if self.session.mode != "hrtem":
            return
        self._record("IFFT 选点")
        added = self.session.toggle_ifft_spot(index)
        self._update_history_actions()
        self.canvas.update()
        self._refresh_hrtem()
        self.say("已加入 IFFT（含对称点）。" if added else "已从 IFFT 移出（含对称点）。")

    def _set_ifft_points(self, points, label):
        self._record(label)
        self.session.hrtem.ifft_points = points
        self.session.hrtem.ifft = None
        self._update_history_actions()
        if self.canvas.view in ("ifft", "compare"):
            self.set_view(self.canvas.view, recompute=True)
        else:
            self.canvas.update()
            self._refresh_hrtem()

    def ifft_use_group(self):
        s = self.selected_set
        if self.session.mode != "hrtem" or s is None:
            self.say("当前没有斑点组。")
            return
        self._set_ifft_points([tuple(self.session.spots[i].xy) for i in s.lattice.members],
                              f"IFFT 使用{s.name}")

    def ifft_use_all(self):
        if self.session.mode == "hrtem":
            self._set_ifft_points(None, "IFFT 使用全部斑点")

    def ifft_mask_changed(self, *args):
        if self._updating or self.session.mode != "hrtem":
            return
        self._record("修改 IFFT 掩膜")
        state = self.session.hrtem
        state.ifft_radius, state.ifft_soft = self.ifft_radius.value(), self.ifft_soft.value()
        state.ifft = None
        self._update_history_actions()
        if self.canvas.view in ("ifft", "compare"):
            self.set_view(self.canvas.view)
        else:
            self.canvas.update()

    def _refresh_hrtem(self):
        hrtem = self.session.mode == "hrtem" and self.session.hrtem is not None
        self.view_bar.setVisible(hrtem)
        self.hrtem_box.setVisible(hrtem)
        for row in range(self.detection_form.rowCount()):
            # only the threshold applies to FFT peak search
            self.detection_form.setRowVisible(row, not hrtem or row == 0)
        ready = self.session.image is not None and self.worker is None
        for name, action in self.mode_actions.items():
            allowed = not ((hrtem and name in SAED_ONLY_MODES) or
                           (not hrtem and name in HRTEM_ONLY_MODES))
            action.setVisible(allowed)
            action.setEnabled(ready and allowed)
        if not hrtem:
            if self.canvas.view != "pattern":
                self.canvas.set_view("pattern")
            return
        state = self.session.hrtem
        previous, self._updating = self._updating, True
        h, w = state.source.gray.shape
        for spin, value, limit in zip(self.roi_fields, state.roi, (w - 32, h - 32, min(w, h))):
            spin.setMaximum(max(limit, 32))
            spin.setValue(int(value))
        self.ifft_radius.setValue(state.ifft_radius)
        self.ifft_soft.setValue(state.ifft_soft)
        self.fft_zoom.setCurrentIndex(max(0, self.fft_zoom.findData(state.zoom_setting)))
        self._updating = previous
        fft = state.fft
        if state.nm_per_px and not state.anisotropic:
            self.fft_info.setText(f"{fft.size} × {fft.size} px · {state.nm_per_px:.5g} nm/px · "
                                  f"显示 {fft.zoom}×\n"
                                  f"FFT 像素 = {1 / fft.px_per_unit:.5g} 1/nm · "
                                  f"Nyquist d = {2 * state.nm_per_px:.4f} nm")
        else:
            self.fft_info.setText(f"{fft.size} × {fft.size} px · 显示 {fft.zoom}× · "
                                  "未定实空间比例，FFT 仅作几何显示")
        points = self.session.ifft_peaks()
        chosen = "全部检测斑点" if state.ifft_points is None else f"{len(points)} 个选点"
        self.ifft_info.setText(f"IFFT：{chosen}（含 Friedel 对称点） · 半径 {state.ifft_radius:g} px"
                               f" · 软边 {state.ifft_soft:g} px")
        self.ifft_button.setEnabled(ready and bool(points))
        self.roi_apply.setEnabled(ready)
        self.view_buttons[self.canvas.view].setChecked(True)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入衍射图", "",
            "衍射图片 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.dm3 *.dm4)")
        if path:
            self.open_image(path)

    def open_image(self, path, background=True, kind="saed"):
        if self.worker:
            return
        if background:
            self._start_worker(path, kind)
        else:
            self._read_settings()
            if kind == "hrtem":
                self.session.open_hrtem(path)
            else:
                self.session.open(path)
            self.session.analyse()
            self.current_set = 0
            self.style.cover_rect = None
            self._clear_history()
            self.canvas.view = "pattern"
            self.refresh(reset=True)

    def _read_settings(self):
        s = self.session
        s.sensitivity, s.min_distance = self.sensitivity.value(), self.min_distance.value()
        s.min_diameter, s.max_diameter = self.min_diameter.value(), self.max_diameter.value()
        s.smoothing = self.smoothing.value()
        s.max_sets = self.max_sets.value()
        s.length_tol, s.scale_tol = self.ratio_tol.value() / 100, self.scale_tol.value() / 100
        s.angle_tol, s.include_forbidden = self.angle_tol.value(), self.forbidden.isChecked()

    def analyse(self):
        if self.session.image is not None and not self.worker:
            if self._reference_dirty:
                self.say("标尺修改尚未应用，请先应用或恢复后再重新识别。")
                return
            if self.max_diameter.value() > 0 and self.min_diameter.value() > self.max_diameter.value():
                self.say("最大直径不能小于最小直径。")
                return
            self._start_worker()

    def _start_worker(self, path=None, kind=None):
        self._analysis_new_image = path is not None
        self._analysis_before = self._snapshot() if path is None else None
        self._read_settings()
        worker = AnalysisWorker(self.session, path, self, kind or self.session.mode)
        self.worker = worker
        worker.result.connect(self._result)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        self._busy(True)
        self.say(("正在计算 FFT 并识别斑点…" if kind == "hrtem" else "正在读取标尺并识别斑点…")
                 if path else "正在重新识别斑点与 CIF 候选…")
        worker.start()

    def _busy(self, busy):
        self.sidebar.setEnabled(not busy)
        self.right_panel.setEnabled(not busy)
        self.canvas.setEnabled(not busy)
        self.view_bar.setEnabled(not busy)
        for action in (self.open_action, self.hrtem_action, self.cif_action, self.run_action,
                       self.export_action, self.csv_action, self.save_action, self.undo_action,
                       self.redo_action):
            action.setEnabled(not busy)

    def _result(self, session):
        self.session = session
        self.current_set = 0
        if self._analysis_new_image:
            self.style.cover_rect = None
            self._clear_history()
            self.canvas.view = "pattern"
        elif self._analysis_before is not None:
            self._push_history("重新识别斑点", self._analysis_before)
        self._analysis_before = None
        self.canvas.selected = None
        self._sync_controls()
        self.refresh(reset=self._analysis_new_image)
        prefix = "FFT " if session.mode == "hrtem" else ""
        self.say(f"{prefix}识别到 {len(session.spots)} 个斑点、{len(session.sets)} 组晶格，候选待确认。")

    def _failed(self, message):
        if self._analysis_before is not None:
            self._restore(self._analysis_before)
        self._analysis_before = None
        QMessageBox.warning(self, "识别未完成", message)
        self.say("识别未完成")

    def _finished(self):
        worker, self.worker = self.worker, None
        self._busy(False)
        self.refresh()
        if worker:
            worker.deleteLater()

    def choose_cifs(self):
        if self.worker:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "导入 CIF", "D:/cif", "CIF (*.cif)")
        if not paths:
            return
        errors = []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for path in paths:
                try:
                    self.session.add_phase(path)
                except Exception as error:
                    errors.append(f"{Path(path).name}: {error}")
            self.session.index_all()
            self.invalidate()
            self._clear_history()
            self.refresh()
        finally:
            QApplication.restoreOverrideCursor()
        if errors:
            QMessageBox.warning(self, "部分 CIF 未导入", "\n".join(errors))

    def remove_phase(self):
        row = self.phase_list.currentRow()
        if row >= 0:
            del self.session.phases[row]
            self.session.index_all()
            self.invalidate()
            self._clear_history()
            self.refresh()

    def phase_changed(self, item):
        if self._updating:
            return
        phase = self.session.phases[self.phase_list.row(item)]
        phase.enabled = item.checkState() == Qt.CheckState.Checked
        self.session.index_all()
        self.invalidate()
        self._clear_history()
        self.refresh()

    def invalidate(self):
        for s in self.session.sets:
            s.confirmed = False

    def apply_calibration(self):
        if self.session.image is None:
            return
        if self.cal_value.value() <= 0:
            self.say("请先输入有效的倒易标定值，或使用两点标尺。")
            return
        if self.session.mode == "hrtem":
            self._record("修改实空间比例")
            self.session.set_real_scale(self.cal_value.value(), "人工输入 nm/px")
            self._reference_dirty = False
            self.refresh()
            self.say(f"实空间比例已应用：{self.cal_value.value():.6g} nm/px。")
            return
        self._record("修改标定")
        self.session.px_per_unit = self.cal_value.value()
        cal = self.session.image.calibration
        cal.px_per_unit = self.session.px_per_unit
        cal.source, cal.confident = "人工校准", True
        cal.conflict, cal.proposed_px_per_unit = False, None
        cal.reference_points, cal.reference_value, cal.reference_unit = None, None, ""
        self._reference_dirty = False
        self.invalidate()
        self._read_settings()
        self.session.index_all()
        self.refresh()

    def manual_scale(self, pixels):
        self.reference_pixels.setValue(pixels)

    def apply_cif_scale(self):
        group = self.selected_set
        if self.worker or group is None or group.candidate is None or self._reference_dirty:
            self.say("请先选择候选，并应用或恢复待修改的标尺。")
            return
        data = group_diagnostics(self.session, group)
        value = data["scale_suggestion"]
        if value is None or not np.isfinite(value) or value <= 0:
            return
        candidate = group.candidate
        if abs(candidate.ratio_err) > self.session.length_tol or \
                abs(candidate.angle_meas - candidate.angle_theo) > self.session.angle_tol:
            self.say("角度或长度比不一致，不能用共同尺度校准消除此冲突。")
            return
        answer = QMessageBox.question(self, "使用 CIF 作为比例参考", 
            f"{candidate.label}\n当前 {self.session.px_per_unit:.5f} → {value:.5f} px/(1/nm)。\n"
            "此操作以该 CIF 晶格参数为参考，会重新索引并清除已有确认。\n"
            "所得比例不能再作为确认该物相的独立证据。是否应用？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._record("应用 CIF 辅助比例")
        if self.session.mode == "hrtem":
            self.session.set_real_scale(value / self.session.hrtem.fft.size,
                                        f"CIF 辅助比例 ({candidate.label})")
            self.refresh()
            self.say("CIF 辅助实空间比例已应用，候选需重新核对；可撤销。")
            return
        cal = self.session.image.calibration
        self.session.px_per_unit = cal.px_per_unit = value
        cal.source = f"CIF 辅助比例 ({candidate.label})"
        cal.confident, cal.conflict, cal.proposed_px_per_unit = True, False, None
        cal.reference_points, cal.reference_value, cal.reference_unit = None, None, ""
        self.session.index_all()
        self.refresh()
        self.say("CIF 辅助比例已应用，候选需重新核对；可撤销。")

    def set_mask(self, rect):
        self.style.cover_rect = rect
        self.canvas.refresh_background()

    def stage_scale(self, points):
        previous = self._updating
        self._updating = True
        for control, value in zip(self.reference_coordinates, np.asarray(points).ravel()):
            control.setValue(float(value))
        self._updating = previous
        self.reference_changed()
        self.say("端点已更新，定标尚未应用。")

    def reference_changed(self, *args):
        if self._updating or self.session.image is None:
            return
        points = np.array([c.value() for c in self.reference_coordinates]).reshape(2, 2)
        length = float(np.linalg.norm(points[1] - points[0]))
        self._updating = True
        self.reference_pixels.setValue(length)
        self._updating = False
        self.canvas.calibration_points = points.tolist()
        self._reference_dirty = True
        self._update_reference_result()
        self._update_history_actions()
        self.canvas.update()

    def reference_length_changed(self, value):
        if self._updating or self.session.image is None:
            return
        points = np.array([c.value() for c in self.reference_coordinates]).reshape(2, 2)
        direction = points[1] - points[0]
        length = np.linalg.norm(direction)
        direction = direction / length if length > 0 else np.array([1., 0.])
        points[1] = points[0] + value * direction
        h, w = self._cal_image().gray.shape
        if (points[1] < 0).any() or points[1, 0] >= w or points[1, 1] >= h:
            self.say("长度超出图像，请先移动起点或调整方向。")
            self._updating = True
            self.reference_pixels.setValue(length)
            self._updating = False
            return
        self.stage_scale(points.tolist())

    def _update_reference_result(self):
        length, value = self.reference_pixels.value(), self.reference_value.value()
        valid = length > 2 and value > 0 and self.session.image is not None
        self.reference_apply.setEnabled(valid and self.worker is None)
        if self.session.mode == "hrtem":
            factor = .1 if self.reference_unit.currentText() != "nm" else 1.
            result = f"{value * factor / length:.6g} nm/px" if valid else "未选定有效端点"
        else:
            factor = 10. if self.reference_unit.currentText() != "1/nm" else 1.
            result = f"{length / (value * factor):.5f} px / (1/nm)" if valid else "未选定有效端点"
        self.reference_result.setText(("待应用：" if self._reference_dirty else "端点换算：") + result)

    def apply_reference(self):
        if self.session.image is None or self.worker:
            return
        before = self._snapshot()
        try:
            self._read_settings()
            points = np.array([c.value() for c in self.reference_coordinates]).reshape(2, 2)
            if self.session.mode == "hrtem":
                self.session.apply_real_reference(points, self.reference_value.value(),
                                                  self.reference_unit.currentText())
            else:
                self.session.apply_reference(points, self.reference_value.value(),
                                             self.reference_unit.currentText())
            self._push_history("应用标尺", before)
            self._reference_dirty = False
            self.refresh()
            if self.session.mode == "hrtem":
                self.say(f"实空间标尺已人工核对并应用：{self.session.hrtem.nm_per_px:.6g} nm/px。")
            else:
                self.say(f"标尺已人工核对并应用：{self.session.px_per_unit:.5f} px/(1/nm)。")
        except ValueError as error:
            self.say(str(error))

    def restore_reference(self):
        self._reference_dirty = False
        self.refresh()
        self.say("已恢复当前定标的参考端点。")

    def _cal_image(self):
        s = self.session
        return s.hrtem.source if s.mode == "hrtem" and s.hrtem is not None else s.image

    def _set_reference_units(self, hrtem):
        units = ["nm", "\u00c5"] if hrtem else ["1/nm", "1/\u00c5"]
        if [self.reference_unit.itemText(i) for i in range(self.reference_unit.count())] != units:
            previous, self._updating = self._updating, True
            self.reference_unit.clear()
            self.reference_unit.addItems(units)
            self._updating = previous

    def _refresh_reference(self, reset=False):
        image = self._cal_image()
        if image is None:
            self.reference_apply.setEnabled(False)
            return
        hrtem = self.session.mode == "hrtem"
        self._set_reference_units(hrtem)
        cal = image.calibration
        value = f"{cal.label_value:g}" if cal.label_value else "未识别"
        pixels = f"{cal.bar['pixels']:.3f} px" if cal.bar else "未识别"
        self.cal_evidence.setText(
            f"原图自动识别：{value} {cal.label_unit}\n"
            f"有效长度：{pixels}\n"
            f"OCR：{cal.label_text or '无文字结果'}")
        evidence = self.cal_evidence.text()
        if hrtem:
            state = self.session.hrtem
            if state.metadata_nm_per_px is not None:
                evidence += f"\nDM 比例：{state.metadata_nm_per_px:.6g} nm/px"
            if state.label_nm_per_px is not None:
                evidence += f"\n图中比例：{state.label_nm_per_px:.6g} nm/px"
        else:
            if cal.metadata_px_per_unit is not None:
                evidence += f"\nDM 比例：{cal.metadata_px_per_unit:.5f}"
            if cal.label_px_per_unit is not None:
                evidence += f"\n图中比例：{cal.label_px_per_unit:.5f}"
            if cal.metadata_px_per_unit is not None or cal.label_px_per_unit is not None:
                evidence += " px/(1/nm)"
        self.cal_evidence.setText(evidence)
        if cal.label_rect:
            x1, y1, x2, y2 = [int(v) for v in cal.label_rect]
            crop = image.gray[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size:
                pixmap = QPixmap.fromImage(qimage_from_array(crop))
                self.cal_thumbnail.setPixmap(pixmap.scaled(
                    max(160, self.sidebar.viewport().width() - 35), 70,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                self.cal_thumbnail.show()
        else:
            self.cal_thumbnail.hide()
        if reset or self._reference_image is not image or not self._reference_dirty:
            self._reference_image = image
            self._reference_dirty = False
            h, w = image.gray.shape
            for i, c in enumerate(self.reference_coordinates):
                c.setMaximum(w - 1 if i % 2 == 0 else h - 1)
            points = self.session.hrtem.reference_points if hrtem else cal.reference_points
            if points is None and cal.bar:
                bar = cal.bar
                start = bar.get("start", bar["x1"])
                points = [[start, bar["y"]], [start + bar["pixels"], bar["y"]]]
            for c, v in zip(self.reference_coordinates, np.asarray(points or [[0, 0], [0, 0]]).ravel()):
                c.setValue(float(v))
            if hrtem:
                state = self.session.hrtem
                self.reference_unit.setCurrentText(state.reference_unit or "nm")
                self.reference_value.setValue(state.reference_value or
                    (cal.label_value if cal.label_unit == "nm" else None) or 5.)
            else:
                selected_unit = cal.reference_unit or cal.label_unit
                self.reference_unit.setCurrentText("1/\u00c5" if selected_unit in ("1/A", "1/\u00c5")
                                                   else "1/nm")
                self.reference_value.setValue(cal.reference_value or
                    (cal.label_value if cal.label_unit.startswith("1/") else None) or 5.)
            coords = np.array([c.value() for c in self.reference_coordinates]).reshape(2, 2)
            self.reference_pixels.setValue(float(np.linalg.norm(coords[1] - coords[0])))
            self.canvas.calibration_points = coords.tolist() if points else None
        self._update_reference_result()

    def geometry_changed(self):
        self._commit_edit("编辑斑点")
        self.session.grouping_report = []
        self.invalidate()
        self.session.index_all()
        self.refresh()

    def add_pair_set(self, first, second):
        before = self._snapshot()
        try:
            self.session.set_from_spots(first, second)
            self._push_history("添加斑点组", before)
            self.current_set = len(self.session.sets) - 1
            self.refresh()
        except ValueError as error:
            self.say(str(error))

    def delete_set(self):
        s = self.selected_set
        if s:
            self._record(f"删除{s.name}")
            self.session.remove_set(s)
            self.current_set = max(0, self.current_set - 1)
            self.refresh()

    # ------------------------------------------------------------ history
    def _snapshot(self):
        """Copy of every user-editable scientific state; phases are shared, not copied."""
        s = self.session
        memo = {id(p): p for p in s.phases}
        cal = s.image.calibration if s.image else None
        return {
            "spots": copy.deepcopy(s.spots, memo), "center": copy.deepcopy(s.center),
            "sets": copy.deepcopy(s.sets, memo), "current_set": self.current_set,
            "px_per_unit": s.px_per_unit,
            "grouping_report": copy.deepcopy(s.grouping_report),
            "settings": {key: getattr(s, key) for key in SETTINGS},
            "calibration": {k: copy.deepcopy(getattr(cal, k)) for k in CALIBRATION_STATE}
            if cal else None,
            "hrtem": copy.deepcopy(s.hrtem_params()), "bar_value": s.bar_value,
        }

    def _restore(self, snap):
        s = self.session
        memo = {id(p): p for p in s.phases}
        if s.hrtem is not None and snap.get("hrtem"):
            s.restore_hrtem_params(copy.deepcopy(snap["hrtem"]))
            s.bar_value = snap["bar_value"]
        s.spots = copy.deepcopy(snap["spots"], memo)
        s.center = copy.deepcopy(snap["center"])
        s.sets = copy.deepcopy(snap["sets"], memo)
        s.px_per_unit = snap["px_per_unit"]
        s.grouping_report = copy.deepcopy(snap["grouping_report"])
        for key, value in snap["settings"].items():
            setattr(s, key, value)
        if s.image and snap["calibration"]:
            for k, v in snap["calibration"].items():
                setattr(s.image.calibration, k, copy.deepcopy(v))
        self.current_set = snap["current_set"]
        self.canvas.selected = None
        self._reference_dirty = False
        self._sync_controls()
        self.refresh()

    def _push_history(self, label, snap):
        self._undo.append((label, snap))
        del self._undo[:-HISTORY_LIMIT]
        self._redo.clear()
        self._pending = None

    def _record(self, label):
        self._push_history(label, self._snapshot())

    def _begin_edit(self):
        self._pending = self._snapshot()

    def _commit_edit(self, label):
        if self._pending is not None:
            self._push_history(label, self._pending)
        self._update_history_actions()

    def _clear_history(self):
        self._undo, self._redo, self._pending = [], [], None

    def _update_history_actions(self):
        ready = self.session.image is not None and self.worker is None
        self.undo_action.setEnabled(ready and (bool(self._undo) or self._reference_dirty))
        self.redo_action.setEnabled(ready and bool(self._redo))
        self.undo_action.setToolTip(
            "撤销：放弃未应用的标尺端点 (Ctrl+Z)" if self._reference_dirty else
            f"撤销：{self._undo[-1][0]} (Ctrl+Z)" if self._undo else "撤销 (Ctrl+Z)")
        self.redo_action.setToolTip(f"重做：{self._redo[-1][0]} (Ctrl+Y)" if self._redo
                                    else "重做 (Ctrl+Y)")

    def undo(self):
        if self.session.image is None or self.worker:
            return
        if self._reference_dirty:
            self.restore_reference()
            self.say("已放弃未应用的标尺端点。")
            return
        if not self._undo:
            return
        label, snap = self._undo.pop()
        self._redo.append((label, self._snapshot()))
        self._restore(snap)
        self.say(f"已撤销：{label}")

    def redo(self):
        if self.session.image is None or self.worker or not self._redo:
            return
        label, snap = self._redo.pop()
        self._undo.append((label, self._snapshot()))
        self._restore(snap)
        self.say(f"已重做：{label}")

    def set_markers_visible(self, visible):
        self.canvas.show_markers = visible
        self.markers_action.setIcon(icon("eye" if visible else "eye-off"))
        self.markers_action.setText("隐藏标记" if visible else "显示标记")
        self.markers_action.setToolTip(
            ("隐藏" if visible else "显示") + "斑点圈、中心、端点及模拟框 (H)")
        self.canvas.update()

    def delete_group_at(self, index):
        if 0 <= index < len(self.session.sets):
            self.current_set = index
            self.delete_set()
            self.say("斑点组已删除，原始斑点保留。")

    def change_set(self, index):
        if not self._updating:
            self.current_set = index
            self.canvas.active_set = index
            self.refresh_details()

    def set_visibility(self, visible):
        if not self._updating and self.selected_set:
            self.selected_set.visible = visible
            self.canvas.update()

    def choose_candidate(self):
        if self._updating or self.selected_set is None:
            return
        row = self.candidate_table.currentRow()
        s = self.selected_set
        if 0 <= row < len(s.candidates):
            s.choice, s.custom, s.labelling, s.confirmed = row, None, 0, False
            self.refresh_details()

    def confirm_set(self):
        if self._reference_dirty:
            self.say("标尺修改未应用，请先应用或恢复定标端点。")
            return
        s = self.selected_set
        if s and s.candidate and self.session.image.calibration.confident:
            s.confirmed = True
            self.refresh_details()
            self.say("当前物相、晶面指数与晶带轴已确认。")
        elif s and s.candidate:
            self.say("标尺识别置信度不足，请先核对标定值并应用标定。")

    def equivalent_changed(self, index):
        if not self._updating and self.selected_set and index >= 0:
            self.selected_set.labelling = index
            self.selected_set.confirmed = False
            self.refresh_details()

    def basis_changed(self, index):
        if not self._updating and self.selected_set:
            self.selected_set.basis = index
            self.canvas.update()

    def title_changed(self):
        if self.selected_set:
            self.selected_set.title = self.title_edit.text()
            self.canvas.update()

    def simulation_changed(self, *args):
        if self._updating:
            return
        if self.selected_set:
            self.selected_set.show_sim = self.simulation_box.isChecked()
        if self.simulation_box.isChecked():
            self.preview_box.setChecked(False)
            self.markers_action.setChecked(True)
        self.style.free_scale = self.free_scale_box.isChecked()
        self.canvas.refresh_simulations()
        self.update_sim_info()

    def residuals_changed(self, checked):
        if self._updating:
            return
        self.style.show_residuals = checked
        self.free_scale_box.setEnabled(not checked)
        if checked:
            self.simulation_box.setChecked(True)
            self.preview_box.setChecked(False)
            self.markers_action.setChecked(True)
        self.canvas.refresh_simulations()
        self.update_sim_info()

    def update_sim_info(self):
        s = self.selected_set
        if not s or not s.candidate or not s.show_sim:
            self.sim_info.setText("")
            return
        free = self.style.free_scale and not self.style.show_residuals
        _, theta, scale = self.session.simulation(s, free_scale=free)
        mode = f"拟合比例 {scale:.4f}×" if free else "固定标尺比例"
        suffix = "；残差始终按固定比例" if self.style.show_residuals else ""
        self.sim_info.setText(f"{mode} · 旋转 {theta:.1f}°{suffix}")

    def display_changed(self, *args):
        if self._updating or not hasattr(self, "canvas"):
            return
        st = self.style
        st.brightness = self.brightness.value()
        st.contrast, st.gamma = self.contrast.value() / 100, self.gamma.value() / 100
        st.log, st.show_spots = self.log_display.isChecked(), self.show_spots.isChecked()
        self.spots_action.setChecked(st.show_spots)
        st.show_center = self.center_box.isChecked()
        st.phase_subscript = self.phase_subscript.isChecked()
        st.chemical_subscripts = self.chemical_subscripts.isChecked()
        st.show_annotations = self.show_annotations.isChecked()
        st.font_size = self.font_size.value()
        st.font_bold = self.font_bold.isChecked()
        st.line_width = self.line_width.value()
        st.replace_bar = self.replace_bar.isChecked()
        st.bar_font_size, st.bar_thickness = self.bar_font_size.value(), self.bar_thickness.value()
        st.publication_preview = self.preview_box.isChecked()
        if st.publication_preview and self.canvas.mode not in ("pan", "labels", "mask"):
            self.mode_actions["pan"].setChecked(True)
            self.canvas.set_mode("pan")
        st.show_calibration = self.show_calibration.isChecked()
        st.simulation_size = self.sim_size.value()
        st.simulation_opacity = self.sim_opacity.value() / 100.
        st.show_dynamic = self.dynamic_box.isChecked()
        if self.session.image:
            self.session.bar_value = self.bar_value.value()
            if self.session.hrtem is not None:
                self.session.hrtem.real_bar_value = self.real_bar_value.value()
        self.canvas.refresh_background()

    def reset_display(self):
        self.brightness.setValue(0)
        self.contrast.setValue(100)
        self.gamma.setValue(100)
        self.log_display.setChecked(False)

    def choose_color(self):
        s = self.selected_set
        if s:
            color = QColorDialog.getColor(QColor(s.color), self, "标注颜色")
            if color.isValid():
                s.color = color.name()
                self.refresh()

    def manual_indices(self):
        s = self.selected_set
        if not s or not self.session.phases or not self.session.px_per_unit:
            return
        found = s.indices()
        initial = (" ".join(str(v) for v in found[0]) + "; " + " ".join(str(v) for v in found[1])
                   if found else "1 0 0; 0 1 0")
        text, ok = QInputDialog.getText(self, "指定 g1 与 g2 的晶面指数", "h k l; h k l", text=initial)
        if not ok:
            return
        try:
            parts = text.split(";")
            pairs = [tuple(int(v) for v in re.findall(r"[+-]?\d+", p)) for p in parts]
            if len(pairs) != 2 or any(len(p) != 3 for p in pairs):
                raise ValueError("请输入两组用分号隔开的三指数，例如 1 -1 0; 0 0 2。")
            self.session.set_custom(s, *pairs)
            s.confirmed = False
            self.refresh_details()
        except ValueError as error:
            QMessageBox.warning(self, "指数无效", str(error))

    def inspect_spot(self, index):
        if index >= len(self.session.spots):
            return
        p = self.session.spots[index]
        text = f"斑点 {index} · ({p.x:.2f}, {p.y:.2f}) px · 积分亮度 {p.intensity:.0f}"
        if p.diameter > 0:
            text += f" · 半峰高直径 {p.diameter:.1f} px"
        if self.session.px_per_unit and self.session.center is not None:
            r = np.linalg.norm(p.xy - self.session.center)
            if r > 1:
                text += f" · d = {self.session.px_per_unit / r:.4f} nm"
        self.spot_info.setText(text)
        self.canvas.selected = index
        self.canvas.update()

    def select_spot_row(self):
        if not self._updating:
            item = self.spots_table.item(self.spots_table.currentRow(), 0)
            if item:
                self.inspect_spot(int(item.text()))

    def select_diagnostic_row(self):
        if not self._updating:
            item = self.diagnostics_table.item(self.diagnostics_table.currentRow(), 0)
            if item:
                self.inspect_spot(int(item.text()))
                self.markers_action.setChecked(True)
                self.preview_box.setChecked(False)

    def refresh_diagnostics(self):
        session, group = self.session, self.selected_set
        self.diagnostics_table.setRowCount(0)
        calibrated = bool(session.px_per_unit and session.image and session.image.calibration.confident)
        self.analysis_status.setText(
            f"定标：{'可用' if calibrated else '待核对'} · 检测：{len(session.spots)} 点 · "
            f"晶格：{len(session.sets)} 组\n"
            f"CIF：{len(group.candidates) if group else 0} 候选 · "
            f"确认：{'已确认' if group and group.confirmed else '待核对'}")
        notes = [r["reason"] + f"（新增 {r['new_spots']} 点）"
                 for r in session.grouping_report if not r["accepted"]]
        self.grouping_notes.setText("；".join(notes[-2:]))
        if group is None:
            self.diagnostics_summary.setText("尚无晶格组，保留原始检测斑点。")
            self.score_details.setText("")
            self.cif_scale_info.setText("")
            self.cif_scale_button.setEnabled(False)
            return None
        data = group_diagnostics(session, group)
        self.diagnostics_summary.setText(
            f"支持 {data['support']} 点 · 相对前组新增 {data['new_support']} 点 · "
            f"共享 {data['shared_support']} 点\n"
            f"覆盖检测点 {data['coverage']:.0%} · 节点占用估计 {data['occupancy']:.0%}\n"
            f"晶格 RMS {data['rms_px']:.2f} px · 最大残差 {data['max_residual_px']:.2f} px\n"
            + (f"实测闭合 {data['closure_px']:.2f} px" if data['closure_px'] is not None else "实测闭合：证据不足"))
        self.score_details.setText("评分分项：" + " · ".join(
            f"{SCORE_LABELS.get(k, k)} {v:.2f}" for k, v in data["cost_parts"].items())
            if data["cost_parts"] else "尚无 CIF 评分")
        for row, record in enumerate(data["spots"]):
            self.diagnostics_table.insertRow(row)
            values = [record["spot"], record["status"],
                      f"{record['residual_px']:.2f}" if record["residual_px"] is not None else "",
                      f"{record['d_error'] * 100:+.2f}" if record["d_error"] is not None else ""]
            for col, value in enumerate(values):
                item = cell(value)
                if record["status"].startswith("偏差"):
                    item.setForeground(QColor("#c0392b"))
                self.diagnostics_table.setItem(row, col, item)
        value = data["scale_suggestion"]
        self.cif_scale_info.setText(f"CIF 比例建议：{value:.5f} px/(1/nm)" if value else "")
        self.cif_scale_button.setEnabled(value is not None and self.worker is None)
        return data

    def refresh(self, reset=False):
        self._updating = True
        session = self.session
        self.phase_list.clear()
        for phase in session.phases:
            item = QListWidgetItem(chemical_text(phase.name))
            item.setToolTip(phase.path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if getattr(phase, "enabled", True)
                               else Qt.CheckState.Unchecked)
            self.phase_list.addItem(item)
        if session.image:
            hrtem = session.mode == "hrtem"
            self._filename_text = ("HRTEM · " if hrtem else "") + session.image.path.name
            self._elide_filename()
            h, w = self._cal_image().gray.shape
            self.zoom_label.setText(f"{w} × {h}")
            cal = session.image.calibration
            self.reference_unit_label.setText("实空间单位" if hrtem else "倒易单位")
            self.cal_value_label.setText("nm / px" if hrtem else "px / (1/nm)")
            self.cal_value.setDecimals(6 if hrtem else 5)
            self.cal_value.setToolTip("每像素对应的实空间长度 nm" if hrtem else
                                      "每 1/nm 对应的像素数，不是 nm/px")
            if hrtem:
                state = session.hrtem
                self.cal_source.setText(f"{state.scale_source} · {state.nm_per_px:.6g} nm/px"
                                        if state.nm_per_px else state.scale_source)
            else:
                self.cal_source.setText(cal.source if session.px_per_unit else "未识别到可靠的倒易标尺")
            self.cal_notes.setText("\n".join(cal.notes) +
                ("\n标定待核对" if not cal.confident else ""))
            if hrtem:
                self.cal_value.setValue(session.hrtem.nm_per_px or 0.)
            elif session.px_per_unit:
                self.cal_value.setValue(session.px_per_unit)
            elif cal.proposed_px_per_unit:
                self.cal_value.setValue(cal.proposed_px_per_unit)
            else:
                self.cal_value.setValue(0.)
            self.bar_value.setValue(session.bar_value or 5.)
            if hrtem:
                self.real_bar_value.setValue(session.hrtem.real_bar_value or 5.)
            self.bar_form.setRowVisible(self.real_bar_label, hrtem)
            self._refresh_reference(reset)
        self.set_combo.clear()
        for s in session.sets:
            self.set_combo.addItem(f"{s.name} · {len(s.lattice.members)} 斑点")
        self.current_set = max(0, min(self.current_set, len(session.sets) - 1)) if session.sets else -1
        if self.current_set >= 0:
            self.set_combo.setCurrentIndex(self.current_set)
        self.canvas.active_set = self.current_set
        image_ready = session.image is not None and self.worker is None
        invalid_sizes = self.max_diameter.value() > 0 and self.min_diameter.value() > self.max_diameter.value()
        self.redetect_button.setEnabled(image_ready and not invalid_sizes)
        self.detection_status.setText(f"已检测 {len(session.spots)} 个斑点" if session.image else "尚未导入图片")
        for a in (self.export_action, self.save_action, self.run_action):
            a.setEnabled(image_ready)
        self.csv_action.setEnabled(image_ready and bool(session.px_per_unit))
        for action in self.mode_actions.values():
            action.setEnabled(image_ready)
        self.add_group_button.setEnabled(image_ready and session.center is not None)
        self.delete_group_button.setEnabled(image_ready and bool(session.sets))
        self._updating = False
        self._refresh_hrtem()
        self.detection_parameters_changed()
        self.canvas.bind(session, self.style, reset)
        self._update_history_actions()
        self.refresh_details()

    def refresh_details(self):
        self._updating = True
        s = self.selected_set
        self.candidate_table.setRowCount(0)
        self.vector_table.setRowCount(0)
        self.spots_table.setRowCount(0)
        self.equivalent_combo.clear()
        diagnostic = self.refresh_diagnostics()
        self.confirm_button.setEnabled(bool(s and s.candidate))
        self.title_edit.setEnabled(s is not None)
        if s is None:
            self.title_edit.clear()
            self.title_edit.setPlaceholderText("")
            self.set_summary.setText(f"{len(self.session.spots)} 个斑点 · 尚无晶格组")
            self.closure.setText("")
            self.match_warning.setText("未找到可靠候选。")
            self.sim_info.setText("")
            self.simulation_box.setChecked(False)
            self.visible_box.setChecked(False)
            self._updating = False
            self.canvas.refresh_simulations()
            return
        self.visible_box.setChecked(s.visible)
        self.simulation_box.setChecked(s.show_sim)
        self.title_edit.setPlaceholderText(s.candidate.phase.name if s.candidate else "")
        self.title_edit.setText(s.title)
        self.basis_combo.setCurrentIndex(s.basis)
        self.set_summary.setText(f"{len(s.lattice.members)} 个斑点 · 晶格拟合 RMS {s.lattice.rms:.2f} px"
                                  f" · {'已确认' if s.confirmed else '待确认'}")
        for row, c in enumerate(s.candidates):
            self.candidate_table.insertRow(row)
            values = [chemical_text(c.phase.name), c.phase.uvw_text(c.zone), f"{c.d1_err * 100:+.1f}",
                      f"{c.d2_err * 100:+.1f}", f"{c.angle_meas - c.angle_theo:+.2f}", f"{c.cost:.2f}"]
            for col, value in enumerate(values):
                self.candidate_table.setItem(row, col, cell(value))
        if s.custom is None and s.candidates:
            self.candidate_table.selectRow(s.choice)
        c = s.candidate
        if c:
            for a, b in s.labellings():
                from .indexing import zone_of
                self.equivalent_combo.addItem(
                    f"{c.phase.uvw_text(zone_of(a, b))}   {c.phase.hkl_text(a)} / {c.phase.hkl_text(b)}")
            self.equivalent_combo.setCurrentIndex(s.labelling)
        rows = self.session.vector_rows(s) if self.session.px_per_unit else []
        for row, r in enumerate(rows):
            self.vector_table.insertRow(row)
            measured = r["observed"]
            values = [r["name"], c.phase.hkl_text(r["hkl"]) if c else "",
                      f"{r['d_meas']:.4f}" if measured else "未检出",
                      f"{r['d_cif']:.4f}" if c else "",
                      f"{r['err'] * 100:+.2f}" if c and measured else "",
                      f"{r['angle_meas']:.2f}" if r["angle_observed"] else "未检出",
                      f"{r['angle_cif']:.2f}" if c else "", "实测" if measured else "晶格推算"]
            for col, value in enumerate(values):
                self.vector_table.setItem(row, col, cell(value))
        for row, r in enumerate(self.session.spot_rows(s) if self.session.px_per_unit else []):
            self.spots_table.insertRow(row)
            kinds = {"allowed": "允许", "dynamic": "动力学可能", "absent": "消光"}
            values = [r["spot"], c.phase.hkl_text(r["hkl"]) if c else str(r["node"]),
                      f"{r['d_meas']:.4f}", f"{r['err'] * 100:+.2f}" if c else "",
                      f"{r['intensity']:.0f}", kinds.get(r.get("kind"), "")]
            for col, value in enumerate(values):
                self.spots_table.setItem(row, col, cell(value))
        closure = next((r for r in rows if "closure_px" in r), None)
        self.closure.setText(f"g1 + g2 = g3 · 实测闭合残差 {closure['closure_px']:.2f} px"
                             f" ({closure['closure_rel'] * 100:.2f}%)" if closure else
                             "g1 + g2 = g3 · 三个实测斑点不齐全，尚不能独立验证闭合")
        warnings = []
        if c:
            warnings.append(f"共同 Δg {c.scale_err * 100:+.2f}% · 动力学候选 {c.dynamic * 100:.0f}%")
            if len(s.candidates) > 1 and s.candidates[1].cost - s.candidates[0].cost < .4:
                warnings.append("前两名评分接近，物相或晶带轴存在歧义。")
            if len(s.lattice.members) < 10 or c.cost > 2:
                warnings.append("匹配证据偏弱，请核对中心及模拟位置。")
        else:
            warnings.append("没有符合容差的 CIF 候选。")
        if diagnostic:
            warnings.extend(diagnostic["warnings"])
        warnings = list(dict.fromkeys(warnings))
        self.match_warning.setText("\n".join(warnings))
        self.match_warning.setMinimumHeight(self.match_warning.fontMetrics().lineSpacing()
                                           * max(1, len(warnings)) + 5)
        self.confirm_button.setText("已确认当前候选" if s.confirmed else "确认当前候选")
        self._updating = False
        self.canvas.refresh_simulations()
        self.canvas.update()
        self.update_sim_info()

    def _export_guard(self):
        if self._reference_dirty:
            self.say("标尺修改未应用，暂未导出。请先核对并应用定标。")
            return False
        if not self.session.px_per_unit:
            QMessageBox.warning(self, "未标定", "请先识别或设置 1/nm 标尺。")
            return False
        pending = [s for s in self.session.sets if s.visible and not s.confirmed]
        if pending:
            return QMessageBox.question(self, "存在未确认结果",
                "可见斑点组仍有未确认候选。继续导出草稿？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
        return True

    def export_image(self):
        if not self.session.image:
            return
        view = self.canvas.view if self.session.mode == "hrtem" else "pattern"
        if view != "pattern":
            return self.export_real_image(view)
        if not self._export_guard():
            return
        suffix = "_fft_indexed.png" if self.session.mode == "hrtem" else "_indexed.png"
        default = str(self.session.image.path.with_name(self.session.image.path.stem + suffix))
        path, _ = QFileDialog.getSaveFileName(self, "导出标定图", default, "PNG (*.png);;TIFF (*.tif)")
        if path:
            try:
                if self.style.replace_bar:
                    length = self.session.bar_value * self.session.px_per_unit
                    if length > self.session.image.gray.shape[1] * .9:
                        raise ValueError("新标尺超过图宽，请减小标尺数值。")
                export_figure(self.session, self.style, path)
                self.say(f"已导出原分辨率标定图 · 600 dpi · {path}")
            except Exception as error:
                QMessageBox.warning(self, "导出失败", str(error))

    def export_real_image(self, view):
        state = self.session.hrtem
        if self._reference_dirty:
            self.say("标尺修改未应用，暂未导出。请先核对并应用定标。")
            return
        if self.style.replace_bar and not state.nm_per_px:
            QMessageBox.warning(self, "未标定", "请先设置 HRTEM 实空间比例，或关闭替换标尺后再导出。")
            return
        names = {"source": "_hrtem.png", "ifft": "_ifft.png", "compare": "_roi_ifft.png"}
        default = str(state.source.path.with_name(state.source.path.stem + names[view]))
        path, _ = QFileDialog.getSaveFileName(self, "导出图像", default, "PNG (*.png);;TIFF (*.tif)")
        if path:
            try:
                if self.style.replace_bar:
                    width = state.source.gray.shape[1] if view == "source" else state.roi[2]
                    if state.real_bar_value / state.nm_per_px > width * .9:
                        raise ValueError("新标尺超过图宽，请减小标尺数值。")
                export_figure(self.session, self.style, path, view=view)
                self.say(f"已导出 {dict(VIEWS)[view]} 图 · 600 dpi · {path}")
            except Exception as error:
                QMessageBox.warning(self, "导出失败", str(error))

    def export_csv(self):
        if not self.session.image or not self.session.px_per_unit:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出数据",
            str(self.session.image.path.with_suffix(".saed.csv")), "CSV (*.csv)")
        if path:
            try:
                self.session.write_csv(path)
                self.say(f"已导出数据：{path}")
            except Exception as error:
                QMessageBox.warning(self, "导出失败", str(error))

    def save_project(self):
        if self.session.image is None or self.worker:
            return
        from .project import save_project
        path, _ = QFileDialog.getSaveFileName(self, "保存项目",
            str(self.session.image.path.with_suffix(".saed.json")), "SAED 项目 (*.saed.json)")
        if path:
            try:
                save_project(path, self.session, self.style)
                self.say(f"已保存项目：{path}")
            except Exception as error:
                QMessageBox.warning(self, "保存失败", str(error))

    def load_project(self):
        if self.worker:
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "SAED 项目 (*.saed.json)")
        if path:
            try:
                from .project import load_project
                self.session, self.style = load_project(path)
                self._clear_history()
                self._sync_controls()
                self.current_set = 0
                self.canvas.view = "pattern"
                self.refresh(reset=True)
            except Exception as error:
                QMessageBox.warning(self, "打开失败", str(error))

    def _sync_controls(self):
        self._updating = True
        self.brightness.setValue(round(self.style.brightness))
        self.contrast.setValue(round(self.style.contrast * 100))
        self.gamma.setValue(round(self.style.gamma * 100))
        self.log_display.setChecked(self.style.log)
        self.font_size.setValue(self.style.font_size)
        self.font_bold.setChecked(self.style.font_bold)
        self.line_width.setValue(self.style.line_width)
        self.bar_font_size.setValue(self.style.bar_font_size)
        self.bar_thickness.setValue(self.style.bar_thickness)
        self.sim_size.setValue(self.style.simulation_size)
        self.sim_opacity.setValue(self.style.simulation_opacity * 100)
        self.dynamic_box.setChecked(self.style.show_dynamic)
        self.preview_box.setChecked(self.style.publication_preview)
        self.show_calibration.setChecked(self.style.show_calibration)
        self.replace_bar.setChecked(self.style.replace_bar)
        self.show_spots.setChecked(self.style.show_spots)
        self.spots_action.setChecked(self.style.show_spots)
        self.center_box.setChecked(self.style.show_center)
        self.phase_subscript.setChecked(self.style.phase_subscript)
        self.chemical_subscripts.setChecked(self.style.chemical_subscripts)
        self.show_annotations.setChecked(self.style.show_annotations)
        self.free_scale_box.setChecked(self.style.free_scale)
        self.residual_box.setChecked(self.style.show_residuals)
        self.free_scale_box.setEnabled(not self.style.show_residuals)
        self.sensitivity.setValue(self.session.sensitivity)
        self.min_distance.setValue(self.session.min_distance)
        self.min_diameter.setValue(self.session.min_diameter)
        self.max_diameter.setValue(self.session.max_diameter)
        self.smoothing.setValue(self.session.smoothing)
        self.max_sets.setValue(self.session.max_sets)
        self.ratio_tol.setValue(self.session.length_tol * 100)
        self.scale_tol.setValue(self.session.scale_tol * 100)
        self.angle_tol.setValue(self.session.angle_tol)
        self.forbidden.setChecked(self.session.include_forbidden)
        self._updating = False

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.say("分析正在进行，请完成后再关闭窗口。")
            event.ignore()
        else:
            event.accept()

    def _elide_filename(self):
        self.filename.setText(self._filename_text)
        self.filename.setToolTip(self._filename_text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "splitter"):
            available = max(self.splitter.width(), self.width() - 48)
            left = 225 if self.width() < 1250 else 255
            right = 385 if self.width() < 1250 else 490
            self.splitter.setSizes([left, max(360, available - left - right - 8), right])
