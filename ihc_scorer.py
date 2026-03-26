#!/usr/bin/env python3
"""
IHC Score Analyzer - 免疫组化评分分析软件
类似ImageJ的IHC评分工具，支持H-Score、阳性率计算、颜色反卷积等功能
"""

import sys
import os
import csv
import platform
import numpy as np
import cv2
import matplotlib
from datetime import datetime

# ─── 配置 matplotlib 中文字体 ─────────────────────────────────────
_system = platform.system()
if _system == "Darwin":
    matplotlib.rcParams["font.sans-serif"] = ["PingFang HK", "Heiti TC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
elif _system == "Windows":
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QGridLayout,
    QTabWidget, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QMessageBox, QSplitter, QCheckBox, QStatusBar,
    QMenuBar, QAction, QToolBar, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon,
    QWheelEvent, QMouseEvent, QKeySequence, QCursor
)
from skimage.color import rgb2hed, hed2rgb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ─── 颜色反卷积矩阵 ───────────────────────────────────────────────
# H-DAB (Hematoxylin - DAB) 标准矩阵
STAIN_VECTORS = {
    "H-DAB": np.array([
        [0.650, 0.704, 0.286],   # Hematoxylin
        [0.268, 0.570, 0.776],   # DAB
        [0.7110272, 0.42318153, 0.5615672]  # Residual
    ]),
    "H-E": np.array([
        [0.644, 0.717, 0.267],   # Hematoxylin
        [0.093, 0.954, 0.283],   # Eosin
        [0.0, 0.0, 0.0]
    ]),
    "H-DAB (skimage)": None,  # 使用skimage内置的HED反卷积
}


class ImageCanvas(QLabel):
    """可缩放、可平移、支持ROI选择的图像显示控件"""
    roi_selected = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")

        self._pixmap = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._dragging = False
        self._drag_start = QPoint()
        self._selecting_roi = False
        self._roi_mode = False
        self._roi_start = QPoint()
        self._roi_end = QPoint()
        self._current_roi = None
        self.setMouseTracking(True)

    def set_image(self, img_array):
        """设置OpenCV格式的图像(BGR或RGB)"""
        if img_array is None:
            self._pixmap = None
            self.clear()
            return
        if len(img_array.shape) == 2:
            h, w = img_array.shape
            bytes_per_line = w
            qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, ch = img_array.shape
            if ch == 3:
                rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if img_array.dtype == np.uint8 else img_array
                bytes_per_line = 3 * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                return
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._fit_to_view()
        self._update_display()

    def set_pixmap_direct(self, pixmap):
        self._pixmap = pixmap
        self._fit_to_view()
        self._update_display()

    def _fit_to_view(self):
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width() - 4, self.height() - 4
        if pw > 0 and ph > 0:
            self._scale = min(vw / pw, vh / ph, 1.0)
            self._offset = QPoint(0, 0)

    def _update_display(self):
        if self._pixmap is None:
            self.setText("拖拽图像到此处或点击 [打开图像]")
            return
        scaled = self._pixmap.scaled(
            int(self._pixmap.width() * self._scale),
            int(self._pixmap.height() * self._scale),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        # 绘制ROI
        if self._current_roi and not self._current_roi.isNull():
            painter = QPainter(scaled)
            pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            r = QRect(
                int(self._current_roi.x() * self._scale),
                int(self._current_roi.y() * self._scale),
                int(self._current_roi.width() * self._scale),
                int(self._current_roi.height() * self._scale),
            )
            painter.drawRect(r)
            painter.end()
        self.setPixmap(scaled)

    def set_roi_mode(self, enabled):
        self._roi_mode = enabled
        self.setCursor(QCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor))

    def clear_roi(self):
        self._current_roi = None
        self._update_display()

    def get_roi(self):
        return self._current_roi

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        new_scale = self._scale * factor
        new_scale = max(0.05, min(new_scale, 20.0))
        self._scale = new_scale
        self._update_display()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self._roi_mode and self._pixmap:
                self._selecting_roi = True
                self._roi_start = self._widget_to_image(event.pos())
                self._roi_end = self._roi_start
            else:
                self._dragging = True
                self._drag_start = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._selecting_roi:
            self._roi_end = self._widget_to_image(event.pos())
            x1, y1 = self._roi_start.x(), self._roi_start.y()
            x2, y2 = self._roi_end.x(), self._roi_end.y()
            self._current_roi = QRect(
                min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
            )
            self._update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self._selecting_roi:
                self._selecting_roi = False
                if self._current_roi and self._current_roi.width() > 5 and self._current_roi.height() > 5:
                    self.roi_selected.emit(self._current_roi)
            self._dragging = False

    def _widget_to_image(self, pos):
        """将控件坐标转换为图像坐标"""
        if self._pixmap is None:
            return QPoint(0, 0)
        # 计算图像在控件中的偏移
        sw = int(self._pixmap.width() * self._scale)
        sh = int(self._pixmap.height() * self._scale)
        ox = (self.width() - sw) // 2
        oy = (self.height() - sh) // 2
        ix = int((pos.x() - ox) / self._scale)
        iy = int((pos.y() - oy) / self._scale)
        ix = max(0, min(ix, self._pixmap.width()))
        iy = max(0, min(iy, self._pixmap.height()))
        return QPoint(ix, iy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap:
            self._update_display()


class HistogramWidget(FigureCanvas):
    """直方图显示控件"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 2.5), dpi=80)
        self.fig.patch.set_facecolor('#2b2b2b')
        super().__init__(self.fig)
        self.setMinimumHeight(180)

    def plot_histogram(self, data, title="", thresholds=None, colors=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_title(title, color='white', fontsize=10)

        if data is not None and len(data) > 0:
            ax.hist(data.ravel(), bins=256, range=(0, 255),
                    color='#4fc3f7', alpha=0.7, edgecolor='none')

            if thresholds:
                color_list = colors or ['#66bb6a', '#ffa726', '#ef5350']
                for i, t in enumerate(thresholds):
                    c = color_list[i] if i < len(color_list) else '#ffffff'
                    ax.axvline(x=t, color=c, linestyle='--', linewidth=1.5)

        ax.set_xlim(0, 255)
        for spine in ax.spines.values():
            spine.set_color('#555')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        self.fig.tight_layout()
        self.draw()


class ScorePieChart(FigureCanvas):
    """饼图显示评分分布"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3, 3), dpi=80)
        self.fig.patch.set_facecolor('#2b2b2b')
        super().__init__(self.fig)
        self.setMinimumHeight(200)

    def plot_scores(self, negative, low_pos, positive, high_pos):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        values = [negative, low_pos, positive, high_pos]
        labels = [f'Negative\n{negative:.1f}%', f'Low+\n{low_pos:.1f}%',
                  f'Positive\n{positive:.1f}%', f'High+\n{high_pos:.1f}%']
        colors_list = ['#42a5f5', '#66bb6a', '#ffa726', '#ef5350']

        non_zero = [(v, l, c) for v, l, c in zip(values, labels, colors_list) if v > 0.1]
        if non_zero:
            vals, labs, cols = zip(*non_zero)
            wedges, texts = ax.pie(vals, labels=labs, colors=cols,
                                    startangle=90, textprops={'color': 'white', 'fontsize': 8})
        ax.set_title('评分分布', color='white', fontsize=10)
        self.fig.tight_layout()
        self.draw()


class BatchResultTable(QTableWidget):
    """批量分析结果表格"""
    def __init__(self):
        super().__init__()
        self.setColumnCount(12)
        self.setHorizontalHeaderLabels([
            'File', 'Neg%', 'Low+%', 'Pos%', 'High+%',
            'Intensity(0-3)', 'Proportion(0-4)', 'IHC Score(0-12)',
            'H-Score', 'Pos Rate%', 'Clinical', 'Area'
        ])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e; color: white;
                gridline-color: #555; alternate-background-color: #2a2a2a;
            }
            QHeaderView::section {
                background-color: #333; color: white;
                padding: 4px; border: 1px solid #555;
            }
        """)

    def add_result(self, results):
        row = self.rowCount()
        self.insertRow(row)
        items = [
            results.get('filename', ''),
            f"{results['negative']:.1f}",
            f"{results['low_pos']:.1f}",
            f"{results['positive']:.1f}",
            f"{results['high_pos']:.1f}",
            str(results['intensity_score']),
            str(results['proportion_score']),
            str(results['ihc_score']),
            f"{results['h_score']:.1f}",
            f"{results['positive_rate']:.1f}",
            results['clinical'],
            results['area_info'],
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            # 临床判定列用颜色标注
            if col == 10:
                if '+++' in text:
                    item.setForeground(QColor('#ef5350'))
                elif '++' in text:
                    item.setForeground(QColor('#ffa726'))
                elif '+' in text:
                    item.setForeground(QColor('#66bb6a'))
                else:
                    item.setForeground(QColor('#42a5f5'))
            self.setItem(row, col, item)


class IHCScorer(QMainWindow):
    """IHC评分分析主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IHC Score Analyzer - 免疫组化评分分析")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # 数据
        self.original_image = None   # BGR
        self.rgb_image = None        # RGB
        self.dab_channel = None      # DAB通道 (灰度, 0-255, 越高越深)
        self.hem_channel = None      # Hematoxylin通道
        self.score_mask = None       # 评分掩膜
        self.current_file = ""
        self.batch_files = []

        self._init_ui()
        self._apply_dark_theme()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #ddd; font-size: 13px; }
            QGroupBox {
                border: 1px solid #555; border-radius: 4px;
                margin-top: 8px; padding-top: 12px;
                font-weight: bold; color: #4fc3f7;
            }
            QGroupBox::title { subcontrol-position: top left; padding: 2px 8px; }
            QPushButton {
                background-color: #333; border: 1px solid #555;
                border-radius: 4px; padding: 6px 14px; color: white;
                min-height: 24px;
            }
            QPushButton:hover { background-color: #444; border-color: #4fc3f7; }
            QPushButton:pressed { background-color: #555; }
            QPushButton#primaryBtn {
                background-color: #1565c0; border-color: #1976d2;
            }
            QPushButton#primaryBtn:hover { background-color: #1976d2; }
            QSlider::groove:horizontal {
                border: 1px solid #555; height: 6px;
                background: #333; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4fc3f7; border: 1px solid #4fc3f7;
                width: 14px; margin: -5px 0; border-radius: 7px;
            }
            QComboBox {
                background-color: #333; border: 1px solid #555;
                border-radius: 4px; padding: 4px 8px; color: white;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #333; color: white; selection-background-color: #1565c0;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #333; border: 1px solid #555;
                border-radius: 4px; padding: 2px 6px; color: white;
            }
            QTabWidget::pane { border: 1px solid #555; background: #1e1e1e; }
            QTabBar::tab {
                background: #333; color: #aaa; padding: 8px 16px;
                border: 1px solid #555; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #1e1e1e; color: #4fc3f7; }
            QStatusBar { background-color: #252525; color: #aaa; }
            QTextEdit {
                background-color: #1e1e1e; color: #ddd;
                border: 1px solid #555; border-radius: 4px;
            }
            QMenuBar { background-color: #252525; color: #ddd; }
            QMenuBar::item:selected { background-color: #333; }
            QMenu { background-color: #333; color: #ddd; border: 1px solid #555; }
            QMenu::item:selected { background-color: #1565c0; }
            QToolBar { background-color: #252525; border-bottom: 1px solid #555; spacing: 4px; }
            QProgressBar {
                border: 1px solid #555; border-radius: 4px;
                text-align: center; color: white; background: #333;
            }
            QProgressBar::chunk { background-color: #1565c0; border-radius: 3px; }
        """)

    def _init_ui(self):
        # ── 菜单栏 ──
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(&F)")

        open_act = QAction("打开图像(&O)", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self.open_image)
        file_menu.addAction(open_act)

        batch_act = QAction("批量打开(&B)", self)
        batch_act.setShortcut("Ctrl+Shift+O")
        batch_act.triggered.connect(self.batch_open)
        file_menu.addAction(batch_act)

        folder_act = QAction("打开文件夹(&D)", self)
        folder_act.setShortcut("Ctrl+D")
        folder_act.triggered.connect(self.open_folder)
        file_menu.addAction(folder_act)

        file_menu.addSeparator()

        export_act = QAction("导出结果(&E)", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self.export_results)
        file_menu.addAction(export_act)

        export_img_act = QAction("保存分析图像(&S)", self)
        export_img_act.setShortcut("Ctrl+S")
        export_img_act.triggered.connect(self.save_analysis_image)
        file_menu.addAction(export_img_act)

        # ── 工具栏 ──
        toolbar = QToolBar("工具栏")
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        btn_open = QPushButton("📂 打开图像")
        btn_open.clicked.connect(self.open_image)
        toolbar.addWidget(btn_open)

        btn_batch = QPushButton("📁 批量打开")
        btn_batch.clicked.connect(self.batch_open)
        toolbar.addWidget(btn_batch)

        btn_folder = QPushButton("📂 打开文件夹")
        btn_folder.clicked.connect(self.open_folder)
        toolbar.addWidget(btn_folder)

        toolbar.addSeparator()

        self.btn_roi = QPushButton("✂️ 选择ROI")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self._toggle_roi_mode)
        toolbar.addWidget(self.btn_roi)

        btn_clear_roi = QPushButton("↩️ 清除ROI")
        btn_clear_roi.clicked.connect(self._clear_roi)
        toolbar.addWidget(btn_clear_roi)

        toolbar.addSeparator()

        btn_analyze = QPushButton("▶ 分析")
        btn_analyze.setObjectName("primaryBtn")
        btn_analyze.clicked.connect(self.analyze_current)
        toolbar.addWidget(btn_analyze)

        btn_batch_analyze = QPushButton("▶▶ 批量分析")
        btn_batch_analyze.setObjectName("primaryBtn")
        btn_batch_analyze.clicked.connect(self.batch_analyze)
        toolbar.addWidget(btn_batch_analyze)

        toolbar.addSeparator()

        btn_export = QPushButton("💾 导出CSV")
        btn_export.clicked.connect(self.export_results)
        toolbar.addWidget(btn_export)

        btn_save_img = QPushButton("🖼 保存图像")
        btn_save_img.clicked.connect(self.save_analysis_image)
        toolbar.addWidget(btn_save_img)

        # ── 主布局 ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── 左侧: 图像显示区 ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.image_tabs = QTabWidget()
        self.canvas_original = ImageCanvas()
        self.canvas_dab = ImageCanvas()
        self.canvas_hem = ImageCanvas()
        self.canvas_score = ImageCanvas()

        self.canvas_original.roi_selected.connect(self._on_roi_selected)
        self.canvas_dab.roi_selected.connect(self._on_roi_selected)

        self.image_tabs.addTab(self.canvas_original, "原始图像")
        self.image_tabs.addTab(self.canvas_dab, "DAB通道")
        self.image_tabs.addTab(self.canvas_hem, "Hematoxylin通道")
        self.image_tabs.addTab(self.canvas_score, "评分结果")

        left_layout.addWidget(self.image_tabs)
        splitter.addWidget(left_widget)

        # ── 右侧: 控制面板 ──
        right_widget = QWidget()
        right_widget.setMaximumWidth(420)
        right_widget.setMinimumWidth(340)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_inner = QWidget()
        right_inner_layout = QVBoxLayout(right_inner)

        # ── 反卷积设置 ──
        deconv_group = QGroupBox("颜色反卷积设置")
        deconv_layout = QGridLayout()

        deconv_layout.addWidget(QLabel("染色方案:"), 0, 0)
        self.stain_combo = QComboBox()
        self.stain_combo.addItems(list(STAIN_VECTORS.keys()))
        self.stain_combo.setCurrentIndex(0)
        deconv_layout.addWidget(self.stain_combo, 0, 1)

        self.chk_auto_balance = QCheckBox("自动白平衡")
        self.chk_auto_balance.setChecked(True)
        deconv_layout.addWidget(self.chk_auto_balance, 1, 0, 1, 2)

        deconv_group.setLayout(deconv_layout)
        right_inner_layout.addWidget(deconv_group)

        # ── 阈值设置 ──
        thresh_group = QGroupBox("阈值设置 (灰度值)")
        thresh_layout = QGridLayout()

        # 阈值说明
        info_label = QLabel(
            "灰度值越低=染色越深\n"
            "High+(0-60) | +(61-120) | Low+(121-180) | -(181-236)"
        )
        info_label.setStyleSheet("color: #888; font-size: 11px;")
        thresh_layout.addWidget(info_label, 0, 0, 1, 3)

        # High Positive / Negative 边界 (灰度60 → DAB 195)
        thresh_layout.addWidget(QLabel("High+ <="), 1, 0)
        self.slider_strong = QSlider(Qt.Horizontal)
        self.slider_strong.setRange(0, 255)
        self.slider_strong.setValue(60)
        self.slider_strong.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_strong, 1, 1)
        self.lbl_strong = QLabel("60")
        self.lbl_strong.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_strong, 1, 2)

        # Positive 边界 (灰度120 → DAB 135)
        thresh_layout.addWidget(QLabel("Positive <="), 2, 0)
        self.slider_moderate = QSlider(Qt.Horizontal)
        self.slider_moderate.setRange(0, 255)
        self.slider_moderate.setValue(120)
        self.slider_moderate.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_moderate, 2, 1)
        self.lbl_moderate = QLabel("120")
        self.lbl_moderate.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_moderate, 2, 2)

        # Low Positive 边界 (灰度180 → DAB 75)
        thresh_layout.addWidget(QLabel("Low+ <="), 3, 0)
        self.slider_weak = QSlider(Qt.Horizontal)
        self.slider_weak.setRange(0, 255)
        self.slider_weak.setValue(180)
        self.slider_weak.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_weak, 3, 1)
        self.lbl_weak = QLabel("180")
        self.lbl_weak.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_weak, 3, 2)

        # 预设
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设:"))
        # (high+, positive, low+) 灰度值
        presets = {
            "标准": (60, 120, 180),
            "严格": (40, 100, 160),
            "宽松": (80, 140, 200),
        }
        for name, (s, m, w) in presets.items():
            btn = QPushButton(name)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked, vals=(s, m, w): self._set_thresholds(*vals))
            preset_layout.addWidget(btn)
        thresh_layout.addLayout(preset_layout, 4, 0, 1, 3)

        # 组织掩膜
        thresh_layout.addWidget(QLabel("背景排除 >="), 5, 0)
        self.slider_tissue = QSlider(Qt.Horizontal)
        self.slider_tissue.setRange(0, 255)
        self.slider_tissue.setValue(236)
        self.slider_tissue.setToolTip("灰度高于此值的区域视为背景(白色)")
        self.slider_tissue.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_tissue, 5, 1)
        self.lbl_tissue = QLabel("236")
        self.lbl_tissue.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_tissue, 5, 2)

        thresh_group.setLayout(thresh_layout)
        right_inner_layout.addWidget(thresh_group)

        # ── DAB直方图 ──
        hist_group = QGroupBox("DAB通道直方图")
        hist_layout = QVBoxLayout()
        self.histogram = HistogramWidget()
        hist_layout.addWidget(self.histogram)
        hist_group.setLayout(hist_layout)
        right_inner_layout.addWidget(hist_group)

        # ── 评分结果 ──
        result_group = QGroupBox("评分结果")
        result_layout = QVBoxLayout()

        self.pie_chart = ScorePieChart()
        result_layout.addWidget(self.pie_chart)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(260)
        self.result_text.setFont(QFont("Courier", 12))
        result_layout.addWidget(self.result_text)

        result_group.setLayout(result_layout)
        right_inner_layout.addWidget(result_group)

        right_inner_layout.addStretch()
        right_scroll.setWidget(right_inner)
        right_layout.addWidget(right_scroll)
        splitter.addWidget(right_widget)

        splitter.setSizes([900, 400])

        # ── 底部: 批量结果标签页 ──
        self.batch_tab = QTabWidget()
        self.batch_table = BatchResultTable()
        self.batch_tab.addTab(self.batch_table, "批量分析结果")
        self.batch_tab.setMaximumHeight(200)
        self.batch_tab.hide()

        outer_layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        outer_layout.addWidget(container)
        outer_layout.addWidget(self.batch_tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        outer_widget = QWidget()
        outer_widget.setLayout(outer_layout)
        self.setCentralWidget(outer_widget)

        # ── 状态栏 ──
        self.statusBar().showMessage("就绪 - 请打开一张IHC染色图像开始分析")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)

    # ─── 文件操作 ─────────────────────────────────────────────────
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开IHC图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.svs);;所有文件 (*)"
        )
        if path:
            self._load_image(path)

    def batch_open(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "批量打开IHC图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;所有文件 (*)"
        )
        if paths:
            self.batch_files = paths
            self._load_image(paths[0])
            self.statusBar().showMessage(f"已加载 {len(paths)} 张图像, 点击[批量分析]开始")

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹", "")
        if not folder:
            return
        IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.svs'}
        paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        if not paths:
            QMessageBox.information(self, "提示", "所选文件夹中未找到图像文件")
            return
        self.batch_files = paths
        self._load_image(paths[0])
        self.statusBar().showMessage(
            f"从文件夹加载 {len(paths)} 张图像, 点击[批量分析]开始")

    @staticmethod
    def _imread_unicode(path):
        """读取图像，支持中文/Unicode路径（兼容Windows）"""
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _load_image(self, path):
        img = self._imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法打开图像:\n{path}")
            return

        self.original_image = img
        self.rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_file = path

        self.canvas_original.set_image(self.rgb_image)
        self.canvas_original.clear_roi()

        # 自动进行颜色反卷积
        self._perform_deconvolution()

        self.setWindowTitle(f"IHC Score Analyzer - {os.path.basename(path)}")
        self.statusBar().showMessage(f"已加载: {path}  |  尺寸: {img.shape[1]}×{img.shape[0]}")

    # ─── 颜色反卷积 ───────────────────────────────────────────────
    def _perform_deconvolution(self):
        if self.rgb_image is None:
            return

        stain_name = self.stain_combo.currentText()
        rgb = self.rgb_image.copy()

        # 白平衡
        if self.chk_auto_balance.isChecked():
            rgb = self._white_balance(rgb)

        if stain_name == "H-DAB (skimage)":
            # 使用skimage的HED反卷积
            img_float = rgb.astype(np.float64) / 255.0
            img_float = np.clip(img_float, 1e-6, 1.0)
            hed = rgb2hed(img_float)
            # H通道
            hem = hed[:, :, 0]
            hem = np.clip(hem, 0, None)
            hem = (hem / (hem.max() + 1e-6) * 255).astype(np.uint8)
            # DAB通道 (E通道在HED中实际对应DAB)
            dab = hed[:, :, 2]
            dab = np.clip(dab, 0, None)
            dab = (dab / (dab.max() + 1e-6) * 255).astype(np.uint8)
        else:
            stain_matrix = STAIN_VECTORS[stain_name]
            hem, dab = self._color_deconvolution(rgb, stain_matrix)

        self.dab_channel = dab
        self.hem_channel = hem

        # 显示通道图像 (使用伪彩色)
        dab_color = cv2.applyColorMap(255 - dab, cv2.COLORMAP_HOT)
        hem_color = cv2.applyColorMap(255 - hem, cv2.COLORMAP_BONE)

        self.canvas_dab.set_image(cv2.cvtColor(dab_color, cv2.COLOR_BGR2RGB))
        self.canvas_hem.set_image(cv2.cvtColor(hem_color, cv2.COLOR_BGR2RGB))

        # 更新直方图
        self._update_histogram()

    def _color_deconvolution(self, rgb_img, stain_matrix):
        """Beer-Lambert颜色反卷积"""
        img = rgb_img.astype(np.float64) / 255.0
        img = np.clip(img, 1e-6, 1.0)

        # 光密度转换
        od = -np.log(img)

        # 归一化染色矩阵
        matrix = stain_matrix.copy().astype(np.float64)
        for i in range(3):
            norm = np.sqrt(np.sum(matrix[i] ** 2))
            if norm > 0:
                matrix[i] /= norm

        # 反卷积
        try:
            inv_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            inv_matrix = np.linalg.pinv(matrix)

        h, w, _ = od.shape
        od_flat = od.reshape(-1, 3)
        stains = od_flat @ inv_matrix.T
        stains = stains.reshape(h, w, 3)

        # 提取各通道并归一化到0-255
        channels = []
        for i in range(2):
            ch = stains[:, :, i]
            ch = np.clip(ch, 0, None)
            # 使用百分位数归一化避免极端值影响
            p99 = np.percentile(ch, 99.5)
            if p99 > 0:
                ch = ch / p99 * 255
            ch = np.clip(ch, 0, 255).astype(np.uint8)
            channels.append(ch)

        return channels[0], channels[1]  # Hematoxylin, DAB

    def _white_balance(self, rgb):
        """简单白平衡"""
        result = rgb.copy().astype(np.float32)
        for i in range(3):
            p95 = np.percentile(result[:, :, i], 95)
            if p95 > 0:
                result[:, :, i] = result[:, :, i] * (255.0 / p95)
        return np.clip(result, 0, 255).astype(np.uint8)

    # ─── 分析 ─────────────────────────────────────────────────────
    def analyze_current(self):
        if self.dab_channel is None:
            QMessageBox.warning(self, "提示", "请先打开一张IHC染色图像")
            return
        results = self._calculate_scores()
        self._display_results(results)
        self._create_score_overlay(results)

    def _calculate_scores(self, dab=None, roi=None):
        """计算IHC评分（基于灰度值标准）"""
        if dab is None:
            dab = self.dab_channel

        # 获取分析区域
        if roi is None:
            roi = self.canvas_original.get_roi()

        if roi and not roi.isNull():
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            x = max(0, x)
            y = max(0, y)
            w = min(w, dab.shape[1] - x)
            h = min(h, dab.shape[0] - y)
            dab_roi = dab[y:y+h, x:x+w]
            area_info = f"ROI({x},{y},{w}x{h})"
        else:
            dab_roi = dab
            area_info = "Full"

        # 灰度值阈值 (从滑块获取)
        t_high = self.slider_strong.value()     # High+ 上界 (默认60)
        t_pos = self.slider_moderate.value()     # Positive 上界 (默认120)
        t_low = self.slider_weak.value()         # Low Positive 上界 (默认180)
        t_tissue = self.slider_tissue.value()    # 背景排除 (默认236)

        # 将DAB通道转为灰度值 (反转: 灰度值低=染色深)
        gray = 255 - dab_roi

        # 背景掩膜 - 排除白色背景
        if self.rgb_image is not None:
            if roi and not roi.isNull():
                x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
                x, y = max(0, x), max(0, y)
                w = min(w, self.rgb_image.shape[1] - x)
                h = min(h, self.rgb_image.shape[0] - y)
                img_gray = cv2.cvtColor(
                    self.rgb_image[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
            else:
                img_gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2GRAY)
            tissue_mask = img_gray < t_tissue
        else:
            tissue_mask = np.ones_like(gray, dtype=bool)

        total_tissue = np.sum(tissue_mask)
        if total_tissue == 0:
            return {
                'negative': 100.0, 'low_pos': 0.0, 'positive': 0.0, 'high_pos': 0.0,
                'h_score': 0.0, 'positive_rate': 0.0,
                'intensity_score': 0, 'proportion_score': 0, 'ihc_score': 0,
                'clinical': 'Negative (-)',
                'total_pixels': int(dab_roi.size), 'tissue_pixels': 0,
                'area_info': area_info
            }

        # 按灰度值分类
        # High Positive: 灰度 0 ~ t_high
        # Positive:      灰度 t_high+1 ~ t_pos
        # Low Positive:  灰度 t_pos+1 ~ t_low
        # Negative:      灰度 t_low+1 ~ t_tissue
        high_mask = tissue_mask & (gray <= t_high)
        pos_mask = tissue_mask & (gray > t_high) & (gray <= t_pos)
        low_mask = tissue_mask & (gray > t_pos) & (gray <= t_low)
        neg_mask = tissue_mask & (gray > t_low)

        n_high = np.sum(high_mask)
        n_pos = np.sum(pos_mask)
        n_low = np.sum(low_mask)
        n_neg = np.sum(neg_mask)

        pct_high = n_high / total_tissue * 100
        pct_pos = n_pos / total_tissue * 100
        pct_low = n_low / total_tissue * 100
        pct_neg = n_neg / total_tissue * 100

        # 阳性率 (所有阳性之和)
        positive_rate = pct_high + pct_pos + pct_low

        # H-Score = 1x(%Low+) + 2x(%Positive) + 3x(%High+), 范围0-300
        h_score = 1 * pct_low + 2 * pct_pos + 3 * pct_high

        # ── 染色强度评分 (Intensity Score, 0-3) ──
        # 取最高强度级别中占比 >10% 的作为主要强度
        if pct_high > 10:
            intensity_score = 3
        elif pct_pos > 10:
            intensity_score = 2
        elif pct_low > 10:
            intensity_score = 1
        else:
            intensity_score = 0

        # ── 阳性比例评分 (Proportion Score, 0-4) ──
        if positive_rate <= 0:
            proportion_score = 0
        elif positive_rate <= 25:
            proportion_score = 1
        elif positive_rate <= 50:
            proportion_score = 2
        elif positive_rate <= 75:
            proportion_score = 3
        else:
            proportion_score = 4

        # ── 最终IHC评分 = 强度 x 比例 (0-12) ──
        ihc_score = intensity_score * proportion_score

        # ── 临床判定 ──
        if ihc_score == 0:
            clinical = "Negative (-)"
        elif ihc_score <= 3:
            clinical = "Weak Positive (+)"
        elif ihc_score <= 6:
            clinical = "Moderate Positive (++)"
        else:
            clinical = "Strong Positive (+++)"

        return {
            'negative': pct_neg, 'low_pos': pct_low,
            'positive': pct_pos, 'high_pos': pct_high,
            'h_score': h_score, 'positive_rate': positive_rate,
            'intensity_score': intensity_score,
            'proportion_score': proportion_score,
            'ihc_score': ihc_score,
            'clinical': clinical,
            'total_pixels': int(dab_roi.size),
            'tissue_pixels': int(total_tissue),
            'area_info': area_info,
            'masks': {
                'negative': neg_mask, 'low_pos': low_mask,
                'positive': pos_mask, 'high_pos': high_mask,
                'tissue': tissue_mask
            }
        }

    def _display_results(self, results):
        """显示评分结果"""
        text = f"""{'='*42}
  IHC Scoring Report
{'='*42}
  File: {os.path.basename(self.current_file)}
  Area: {results['area_info']}
  Tissue: {results['tissue_pixels']:,} px
{'_'*42}
  Negative (181-236): {results['negative']:6.1f}%
  Low Positive(121-180): {results['low_pos']:6.1f}%
  Positive   (61-120): {results['positive']:6.1f}%
  High Positive (0-60): {results['high_pos']:6.1f}%
{'_'*42}
  Intensity Score:   {results['intensity_score']}  (0-3)
  Proportion Score:  {results['proportion_score']}  (0-4)
{'_'*42}
  H-Score:       {results['h_score']:6.1f} / 300
  Positive Rate: {results['positive_rate']:6.1f}%
  IHC Score:     {results['ihc_score']:>2d}    (0-12)
{'='*42}
  >> {results['clinical']}
{'='*42}"""

        self.result_text.setPlainText(text)

        # 更新饼图
        self.pie_chart.plot_scores(
            results['negative'], results['low_pos'],
            results['positive'], results['high_pos']
        )

        self.statusBar().showMessage(
            f"IHC={results['ihc_score']} | "
            f"{results['clinical']} | "
            f"H-Score={results['h_score']:.1f}"
        )

    def _create_score_overlay(self, results):
        """创建评分叠加图像"""
        if 'masks' not in results or self.rgb_image is None:
            return

        masks = results['masks']
        roi = self.canvas_original.get_roi()

        if roi and not roi.isNull():
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            x, y = max(0, x), max(0, y)
            w = min(w, self.rgb_image.shape[1] - x)
            h = min(h, self.rgb_image.shape[0] - y)
            overlay = self.rgb_image[y:y+h, x:x+w].copy()
        else:
            overlay = self.rgb_image.copy()

        # 半透明叠加颜色
        alpha = 0.45

        # Negative - 蓝色
        overlay[masks['negative']] = (
            overlay[masks['negative']] * (1 - alpha) +
            np.array([66, 165, 245]) * alpha
        ).astype(np.uint8)

        # Low Positive - 绿色
        overlay[masks['low_pos']] = (
            overlay[masks['low_pos']] * (1 - alpha) +
            np.array([102, 187, 106]) * alpha
        ).astype(np.uint8)

        # Positive - 橙色
        overlay[masks['positive']] = (
            overlay[masks['positive']] * (1 - alpha) +
            np.array([255, 167, 38]) * alpha
        ).astype(np.uint8)

        # High Positive - 红色
        overlay[masks['high_pos']] = (
            overlay[masks['high_pos']] * (1 - alpha) +
            np.array([239, 83, 80]) * alpha
        ).astype(np.uint8)

        # 背景 - 灰暗
        bg = ~masks['tissue']
        overlay[bg] = (overlay[bg] * 0.3).astype(np.uint8)

        self.score_mask = overlay
        self.canvas_score.set_image(overlay)
        self.image_tabs.setCurrentIndex(3)  # 切换到评分结果标签

    def _update_histogram(self):
        """更新灰度直方图"""
        if self.dab_channel is None:
            return
        roi = self.canvas_original.get_roi()
        if roi and not roi.isNull():
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            x, y = max(0, x), max(0, y)
            data = 255 - self.dab_channel[y:y+h, x:x+w]
        else:
            data = 255 - self.dab_channel

        # 灰度值阈值线: High+, Positive, Low+
        thresholds = [
            self.slider_strong.value(),   # High+ boundary
            self.slider_moderate.value(),  # Positive boundary
            self.slider_weak.value(),      # Low+ boundary
        ]
        self.histogram.plot_histogram(
            data, "Grayscale Distribution",
            thresholds, colors=['#ef5350', '#ffa726', '#66bb6a']
        )

    # ─── 阈值变化 ─────────────────────────────────────────────────
    def _on_threshold_changed(self):
        # 确保阈值逻辑: high+ < positive < low+ < tissue
        s = self.slider_strong.value()   # High+
        m = self.slider_moderate.value()  # Positive
        w = self.slider_weak.value()      # Low+

        if m <= s:
            m = s + 1
            self.slider_moderate.blockSignals(True)
            self.slider_moderate.setValue(m)
            self.slider_moderate.blockSignals(False)
        if w <= m:
            w = m + 1
            self.slider_weak.blockSignals(True)
            self.slider_weak.setValue(w)
            self.slider_weak.blockSignals(False)

        self.lbl_strong.setText(str(self.slider_strong.value()))
        self.lbl_moderate.setText(str(self.slider_moderate.value()))
        self.lbl_weak.setText(str(self.slider_weak.value()))
        self.lbl_tissue.setText(str(self.slider_tissue.value()))

        self._update_histogram()

    def _set_thresholds(self, high, pos, low):
        self.slider_strong.setValue(high)
        self.slider_moderate.setValue(pos)
        self.slider_weak.setValue(low)

    # ─── ROI ──────────────────────────────────────────────────────
    def _toggle_roi_mode(self, checked):
        self.canvas_original.set_roi_mode(checked)
        self.canvas_dab.set_roi_mode(checked)
        if checked:
            self.statusBar().showMessage("ROI模式: 在图像上拖拽选择分析区域")

    def _on_roi_selected(self, roi):
        self.btn_roi.setChecked(False)
        self._update_histogram()
        self.statusBar().showMessage(
            f"已选择ROI: ({roi.x()}, {roi.y()}) - "
            f"{roi.width()}×{roi.height()}"
        )

    def _clear_roi(self):
        self.canvas_original.clear_roi()
        self.canvas_dab.clear_roi()
        self._update_histogram()
        self.statusBar().showMessage("已清除ROI选区")

    # ─── 批量分析 ──────────────────────────────────────────────────
    def batch_analyze(self):
        if not self.batch_files:
            QMessageBox.information(self, "提示", "请先通过[批量打开]加载图像文件")
            return

        self.batch_tab.show()
        # 清空旧结果
        self.batch_table.setRowCount(0)

        self.progress_bar.show()
        self.progress_bar.setRange(0, len(self.batch_files))

        for i, path in enumerate(self.batch_files):
            self.progress_bar.setValue(i)
            QApplication.processEvents()

            img = self._imread_unicode(path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.chk_auto_balance.isChecked():
                rgb = self._white_balance(rgb)

            stain_name = self.stain_combo.currentText()
            if stain_name == "H-DAB (skimage)":
                img_float = rgb.astype(np.float64) / 255.0
                img_float = np.clip(img_float, 1e-6, 1.0)
                hed = rgb2hed(img_float)
                dab = hed[:, :, 2]
                dab = np.clip(dab, 0, None)
                dab = (dab / (dab.max() + 1e-6) * 255).astype(np.uint8)
            else:
                stain_matrix = STAIN_VECTORS[stain_name]
                _, dab = self._color_deconvolution(rgb, stain_matrix)

            # 临时设置图像用于计算
            old_rgb = self.rgb_image
            self.rgb_image = rgb
            results = self._calculate_scores(dab=dab, roi=None)
            self.rgb_image = old_rgb

            results['filename'] = os.path.basename(path)
            self.batch_table.add_result(results)

        self.progress_bar.setValue(len(self.batch_files))
        self.progress_bar.hide()
        self.statusBar().showMessage(
            f"批量分析完成: {len(self.batch_files)} 张图像"
        )

    # ─── 导出 ─────────────────────────────────────────────────────
    def export_results(self):
        if self.batch_table.rowCount() == 0:
            # 只有单张图像结果
            if self.dab_channel is None:
                QMessageBox.information(self, "提示", "没有可导出的结果")
                return
            results = self._calculate_scores()
            path, _ = QFileDialog.getSaveFileName(
                self, "导出结果", f"ihc_result_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "CSV文件 (*.csv)"
            )
            if path:
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'File', 'Neg%', 'Low+%', 'Pos%', 'High+%',
                        'Intensity(0-3)', 'Proportion(0-4)', 'IHC Score(0-12)',
                        'H-Score', 'Pos Rate%', 'Clinical',
                        'Tissue Pixels', 'Area'
                    ])
                    writer.writerow([
                        os.path.basename(self.current_file),
                        f"{results['negative']:.2f}",
                        f"{results['low_pos']:.2f}",
                        f"{results['positive']:.2f}",
                        f"{results['high_pos']:.2f}",
                        results['intensity_score'],
                        results['proportion_score'],
                        results['ihc_score'],
                        f"{results['h_score']:.2f}",
                        f"{results['positive_rate']:.2f}",
                        results['clinical'],
                        results['tissue_pixels'],
                        results['area_info'],
                    ])
                self.statusBar().showMessage(f"结果已导出: {path}")
        else:
            # 批量结果
            path, _ = QFileDialog.getSaveFileName(
                self, "导出批量结果",
                f"ihc_batch_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "CSV文件 (*.csv)"
            )
            if path:
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    headers = []
                    for col in range(self.batch_table.columnCount()):
                        headers.append(
                            self.batch_table.horizontalHeaderItem(col).text())
                    writer.writerow(headers)
                    for row in range(self.batch_table.rowCount()):
                        row_data = []
                        for col in range(self.batch_table.columnCount()):
                            item = self.batch_table.item(row, col)
                            row_data.append(item.text() if item else "")
                        writer.writerow(row_data)
                self.statusBar().showMessage(f"批量结果已导出: {path}")

    def save_analysis_image(self):
        if self.score_mask is None:
            QMessageBox.information(self, "提示", "请先执行分析")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存分析图像",
            f"ihc_analysis_{datetime.now():%Y%m%d_%H%M%S}.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)"
        )
        if path:
            save_img = cv2.cvtColor(self.score_mask, cv2.COLOR_RGB2BGR)
            ext = os.path.splitext(path)[1]
            result, buf = cv2.imencode(ext, save_img)
            if result:
                buf.tofile(path)
            self.statusBar().showMessage(f"分析图像已保存: {path}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("IHC Score Analyzer")

    # 高DPI支持
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = IHCScorer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
