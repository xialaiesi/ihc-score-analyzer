#!/usr/bin/env python3
"""
FAST IHC Analyzer - Fast Immunohistochemistry Analysis Software.
Supporting H-Score, positive rate calculation, and batch analysis.
"""

import sys
import os
import csv
import platform
import numpy as np
import cv2
from PIL import Image
import matplotlib
from datetime import datetime

# ─── Configure matplotlib Chinese fonts ───────────────────────────
_system = platform.system()
if _system == "Darwin":
    matplotlib.rcParams["font.sans-serif"] = ["PingFang HK", "Heiti TC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
elif _system == "Windows":
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.serif"] = ["Times New Roman"]
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QGridLayout,
    QTabWidget, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QMessageBox, QSplitter, QCheckBox, QStatusBar,
    QMenuBar, QAction, QToolBar, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar
)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon,
    QWheelEvent, QMouseEvent, QKeySequence, QCursor
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageCanvas(QLabel):
    """Zoomable, pannable image display widget with ROI selection."""
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

    def set_image(self, img_array, is_rgb=False):
        """Set image; is_rgb=True means the input is already in RGB format."""
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
                if is_rgb:
                    rgb = np.ascontiguousarray(img_array)
                else:
                    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                return
        self._pixmap = QPixmap.fromImage(qimg.copy())
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
        # Draw ROI overlay
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
        """Convert widget coordinates to image coordinates."""
        if self._pixmap is None:
            return QPoint(0, 0)
        # Calculate image offset within the widget
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
    """Histogram display widget."""
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
    """Pie chart for score distribution display."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3, 3), dpi=80)
        self.fig.patch.set_facecolor('#2b2b2b')
        super().__init__(self.fig)
        self.setMinimumHeight(200)

    def plot_scores(self, negative, low_pos, positive, high_pos, lang='zh'):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        values = [negative, low_pos, positive, high_pos]
        if lang == 'zh':
            labels = [f'阴性\n{negative:.1f}%', f'弱阳性\n{low_pos:.1f}%',
                      f'阳性\n{positive:.1f}%', f'强阳性\n{high_pos:.1f}%']
            title = '评分分布'
        else:
            labels = [f'Neg\n{negative:.1f}%', f'Low+\n{low_pos:.1f}%',
                      f'Pos\n{positive:.1f}%', f'High+\n{high_pos:.1f}%']
            title = 'Score Distribution'
        colors_list = ['#42a5f5', '#66bb6a', '#ffa726', '#ef5350']

        non_zero = [(v, l, c) for v, l, c in zip(values, labels, colors_list) if v > 0.1]
        if non_zero:
            vals, labs, cols = zip(*non_zero)
            wedges, texts = ax.pie(vals, labels=labs, colors=cols,
                                    startangle=90, textprops={'color': 'white', 'fontsize': 8})
        ax.set_title(title, color='white', fontsize=10)
        self.fig.tight_layout()
        self.draw()


class BatchResultTable(QTableWidget):
    """Batch analysis results table."""
    def __init__(self):
        super().__init__()
        self.setColumnCount(11)
        self.setHorizontalHeaderLabels([
            '序号', '图片名称', '总像素', '高强阳(%)', '中阳(%)',
            '低阳(%)', '阴性(%)', '临床判定', '强度评分', '比例评分', 'IHC评分'
        ])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
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
            str(row + 1),
            results.get('filename', ''),
            f"{results['total_pixels']:,}",
            f"{results['high_pos']:.2f}",
            f"{results['positive']:.2f}",
            f"{results['low_pos']:.2f}",
            f"{results['negative']:.2f}",
            results['clinical'],
            str(results['intensity_score']),
            str(results['proportion_score']),
            str(results['ihc_score']),
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            # Color-code the clinical determination column
            if col == 7:
                if 'Positive' in text or '阳性' in text:
                    item.setForeground(QColor('#ffa726'))
                else:
                    item.setForeground(QColor('#42a5f5'))
            self.setItem(row, col, item)


class IHCScorer(QMainWindow):
    """IHC scoring analysis main window."""
    CLINICAL_POSITIVE_THRESHOLD = 5.0

    # ── Bilingual text ──
    LANG_ZH = {
        'title': 'FAST IHC Analyzer - 快速免疫组化分析',
        'open': '打开图像(&O)', 'open_folder': '打开文件夹(&D)',
        'export': '导出结果(&E)', 'save_img': '保存分析图像(&S)',
        'toolbar_open': '打开图像', 'toolbar_folder': '打开文件夹',
        'toolbar_prev': '上一张', 'toolbar_next': '下一张',
        'toolbar_roi': '选择ROI', 'toolbar_clear_roi': '清除ROI',
        'toolbar_analyze': '分析', 'toolbar_batch_analyze': '批量分析',
        'toolbar_export': '导出CSV', 'toolbar_save': '保存图像',
        'grp_deconv': '检测设置',
        'detect_info': '使用 HSV 色彩空间检测阳性区域\nHue: 0-20 | Sat >= 50 | Val >= 50',
        'stain_label': '染色方案:',
        'auto_wb': '自动白平衡', 'preset_default': '默认',
        'grp_thresh': '阈值设置 (灰度值)',
        'lbl_high': '强阳性 <=', 'lbl_pos': '阳性 <=', 'lbl_low': '弱阳性 <=',
        'preset_label': '预设:', 'preset_std': '标准', 'preset_strict': '严格', 'preset_loose': '宽松',
        'bg_label': '背景排除 >=',
        'grp_hist': 'DAB通道直方图', 'grp_result': '评分结果',
        'tab_original': '原始图像', 'tab_dab': 'DAB通道',
        'tab_hem': 'Hematoxylin通道', 'tab_score': '评分结果',
        'batch_tab': '批量分析结果',
        'table_headers': ['序号', '图片名称', '总像素', '高强阳(%)', '中阳(%)',
                          '低阳(%)', '阴性(%)', '临床判定', '强度评分', '比例评分', 'IHC评分'],
        'status_ready': '就绪 - 请打开一张IHC染色图像开始分析',
        'lang_switch': 'English',
    }
    LANG_EN = {
        'title': 'FAST IHC Analyzer',
        'open': 'Open Image(&O)', 'open_folder': 'Open Folder(&D)',
        'export': 'Export Results(&E)', 'save_img': 'Save Analysis Image(&S)',
        'toolbar_open': 'Open', 'toolbar_folder': 'Open Folder',
        'toolbar_prev': 'Prev', 'toolbar_next': 'Next',
        'toolbar_roi': 'Select ROI', 'toolbar_clear_roi': 'Clear ROI',
        'toolbar_analyze': 'Analyze', 'toolbar_batch_analyze': 'Batch Analyze',
        'toolbar_export': 'Export CSV', 'toolbar_save': 'Save Image',
        'grp_deconv': 'Detection',
        'detect_info': 'HSV color space positive detection\nHue: 0-20 | Sat >= 50 | Val >= 50',
        'stain_label': 'Stain:',
        'auto_wb': 'Auto White Balance', 'preset_default': 'Default',
        'grp_thresh': 'Threshold (Grayscale)',
        'lbl_high': 'High+ <=', 'lbl_pos': 'Positive <=', 'lbl_low': 'Low+ <=',
        'preset_label': 'Preset:', 'preset_std': 'Standard', 'preset_strict': 'Strict', 'preset_loose': 'Loose',
        'bg_label': 'Background >=',
        'grp_hist': 'DAB Histogram', 'grp_result': 'Scoring Result',
        'tab_original': 'Original', 'tab_dab': 'DAB Channel',
        'tab_hem': 'Hematoxylin', 'tab_score': 'Score Overlay',
        'batch_tab': 'Batch Results',
        'table_headers': ['No.', 'Filename', 'Pixels', 'High+(%)', 'Pos(%)',
                          'Low+(%)', 'Neg(%)', 'Clinical', 'Intensity', 'Proportion', 'IHC Score'],
        'status_ready': 'Ready - Open an IHC stained image to begin',
        'lang_switch': '中文',
    }

    def __init__(self):
        super().__init__()
        self.lang = self.LANG_ZH
        self.setWindowTitle(self.lang['title'])
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # Data
        self.original_image = None   # BGR
        self.rgb_image = None        # RGB
        self.dab_channel = None      # DAB channel (masked image grayscale)
        self.hem_channel = None      # Hematoxylin channel (preprocessed grayscale)
        self.score_mask = None       # Score overlay mask
        self.current_file = ""
        self.batch_files = []
        self.current_index = -1           # Current image index
        self.batch_results_cache = {}     # {index: results_dict} batch analysis result cache
        self.batch_image_cache = {}       # {index: (rgb, preprocessed, masked, mask, dab_gray, pos_ratio)}
        # TIFF-workflow additions
        self.preprocessed_image = None  # CLAHE-preprocessed RGB image
        self.masked_image = None        # HSV-masked image
        self.hsv_mask = None            # HSV binary mask
        self.positive_ratio = 0.0       # Positive pixel ratio
        self.hsv_params = {
            'hue_low': 0, 'hue_high': 20,
            'saturation_low': 50, 'value_low': 50
        }

        self._init_ui()
        self._apply_dark_theme()

        # Set window icon (supports PyInstaller bundled environment)
        self.setWindowIcon(self._load_app_icon())

    @staticmethod
    def _get_resource_path(filename):
        """Get absolute path to resource, works for dev and PyInstaller."""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, filename)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    def _load_app_icon(self):
        """Load application icon from bundled or local resource."""
        for name in ('icon.ico', 'icon.png'):
            path = self._get_resource_path(name)
            if os.path.exists(path):
                return QIcon(path)
        return QIcon()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #ddd; font-size: 13px; font-family: "Times New Roman", "PingFang SC", "Microsoft YaHei", sans-serif; }
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
            QMenuBar { background-color: #252525; color: #ddd; font-size: 18px; padding: 2px; }
            QMenuBar::item:selected { background-color: #333; }
            QMenu { background-color: #333; color: #ddd; border: 1px solid #555; font-size: 17px; }
            QMenu::item:selected { background-color: #1565c0; }
            QMenu::item { padding: 6px 20px; }
            QToolBar { background-color: #252525; border-bottom: 1px solid #555; spacing: 4px; font-size: 16px; }
            QToolBar QPushButton { font-size: 16px; padding: 6px 14px; }
            QProgressBar {
                border: 1px solid #555; border-radius: 4px;
                text-align: center; color: white; background: #333;
            }
            QProgressBar::chunk { background-color: #1565c0; border-radius: 3px; }
        """)

    def _init_ui(self):
        # ── Menu bar ──
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(&F)")

        self.act_open = QAction("打开图像(&O)", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.triggered.connect(self.open_image)
        file_menu.addAction(self.act_open)

        self.act_folder = QAction("打开文件夹(&D)", self)
        self.act_folder.setShortcut("Ctrl+D")
        self.act_folder.triggered.connect(self.open_folder)
        file_menu.addAction(self.act_folder)

        file_menu.addSeparator()

        self.act_export = QAction("导出结果(&E)", self)
        self.act_export.setShortcut("Ctrl+E")
        self.act_export.triggered.connect(self.export_results)
        file_menu.addAction(self.act_export)

        self.act_save_img = QAction("保存分析图像(&S)", self)
        self.act_save_img.setShortcut("Ctrl+S")
        self.act_save_img.triggered.connect(self.save_analysis_image)
        file_menu.addAction(self.act_save_img)

        # ── Toolbar ──
        toolbar = QToolBar("工具栏")
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        self.btn_open = QPushButton("📂 打开图像")
        self.btn_open.clicked.connect(self.open_image)
        toolbar.addWidget(self.btn_open)

        self.btn_folder = QPushButton("📂 打开文件夹")
        self.btn_folder.clicked.connect(self.open_folder)
        toolbar.addWidget(self.btn_folder)

        toolbar.addSeparator()

        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.clicked.connect(self._prev_image)
        toolbar.addWidget(self.btn_prev)

        self.lbl_image_index = QLabel("")
        self.lbl_image_index.setStyleSheet("color: #aaa; padding: 0 6px;")
        toolbar.addWidget(self.lbl_image_index)

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.clicked.connect(self._next_image)
        toolbar.addWidget(self.btn_next)

        toolbar.addSeparator()

        self.btn_roi = QPushButton("[+] 选择ROI")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self._toggle_roi_mode)
        toolbar.addWidget(self.btn_roi)

        self.btn_clear_roi = QPushButton("[x] 清除ROI")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        toolbar.addWidget(self.btn_clear_roi)

        toolbar.addSeparator()

        self.btn_analyze = QPushButton("▶ 分析")
        self.btn_analyze.setObjectName("primaryBtn")
        self.btn_analyze.clicked.connect(self.analyze_current)
        toolbar.addWidget(self.btn_analyze)

        self.btn_batch_analyze = QPushButton("▶▶ 批量分析")
        self.btn_batch_analyze.setObjectName("primaryBtn")
        self.btn_batch_analyze.clicked.connect(self.batch_analyze)
        toolbar.addWidget(self.btn_batch_analyze)

        toolbar.addSeparator()

        self.btn_export = QPushButton("💾 导出CSV")
        self.btn_export.clicked.connect(self.export_results)
        toolbar.addWidget(self.btn_export)

        self.btn_save_img = QPushButton("🖼 保存图像")
        self.btn_save_img.clicked.connect(self.save_analysis_image)
        toolbar.addWidget(self.btn_save_img)

        toolbar.addSeparator()

        self.btn_lang = QPushButton("🌐 English")
        self.btn_lang.clicked.connect(self._toggle_language)
        toolbar.addWidget(self.btn_lang)

        # ── Main layout ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left side: image display area ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Image area + side navigation arrows
        image_area = QWidget()
        image_h_layout = QHBoxLayout(image_area)
        image_h_layout.setContentsMargins(0, 0, 0, 0)
        image_h_layout.setSpacing(0)

        self.btn_prev_side = QPushButton("◀")
        self.btn_prev_side.setFixedWidth(32)
        self.btn_prev_side.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.btn_prev_side.clicked.connect(self._prev_image)
        self.btn_prev_side.setStyleSheet("""
            QPushButton {
                background-color: rgba(30,30,30,180); color: #aaa;
                border: none; font-size: 18px; border-radius: 4px;
            }
            QPushButton:hover { background-color: rgba(60,60,60,220); color: white; }
        """)
        image_h_layout.addWidget(self.btn_prev_side)

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
        image_h_layout.addWidget(self.image_tabs)

        self.btn_next_side = QPushButton("▶")
        self.btn_next_side.setFixedWidth(32)
        self.btn_next_side.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.btn_next_side.clicked.connect(self._next_image)
        self.btn_next_side.setStyleSheet("""
            QPushButton {
                background-color: rgba(30,30,30,180); color: #aaa;
                border: none; font-size: 18px; border-radius: 4px;
            }
            QPushButton:hover { background-color: rgba(60,60,60,220); color: white; }
        """)
        image_h_layout.addWidget(self.btn_next_side)

        left_layout.addWidget(image_area)
        splitter.addWidget(left_widget)

        # ── Right side: control panel ──
        right_widget = QWidget()
        right_widget.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_inner = QWidget()
        right_inner_layout = QVBoxLayout(right_inner)

        # Hidden controls (kept for language-switch compatibility)
        self.grp_deconv = QGroupBox()
        self.lbl_stain = QLabel()
        self.stain_combo = QComboBox()
        self.chk_auto_balance = QCheckBox()
        self.chk_auto_balance.setChecked(True)
        self.lbl_detect_info = QLabel()
        self.grp_thresh = QGroupBox()
        self.threshold_info_label = QLabel()
        self.lbl_high_tag = QLabel()
        self.lbl_pos_tag = QLabel()
        self.lbl_low_tag = QLabel()
        self.lbl_bg_tag = QLabel()
        self.lbl_preset = QLabel()
        self.lbl_strong = QLabel("160")
        self.lbl_moderate = QLabel("100")
        self.lbl_weak = QLabel("40")
        self.lbl_tissue = QLabel("236")
        self.slider_strong = QSlider(Qt.Horizontal)
        self.slider_strong.setRange(0, 255)
        self.slider_strong.setValue(160)
        self.slider_moderate = QSlider(Qt.Horizontal)
        self.slider_moderate.setRange(0, 255)
        self.slider_moderate.setValue(100)
        self.slider_weak = QSlider(Qt.Horizontal)
        self.slider_weak.setRange(0, 255)
        self.slider_weak.setValue(40)
        self.slider_tissue = QSlider(Qt.Horizontal)
        self.slider_tissue.setRange(0, 255)
        self.slider_tissue.setValue(236)
        self.btn_preset_default = QPushButton()
        self.btn_preset_std = QPushButton()
        self.btn_preset_strict = QPushButton()
        self.btn_preset_loose = QPushButton()
        self.grp_hist = QGroupBox()
        self.histogram = HistogramWidget()

        # ── Scoring results (fills the entire right panel) ──
        self.grp_result = QGroupBox("评分结果")
        result_layout = QVBoxLayout()

        self.pie_chart = ScorePieChart()
        self.pie_chart.setMinimumHeight(250)
        result_layout.addWidget(self.pie_chart)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Times New Roman", 13))
        result_layout.addWidget(self.result_text, 1)  # stretch=1 auto-fill

        self.grp_result.setLayout(result_layout)
        right_inner_layout.addWidget(self.grp_result, 1)  # stretch=1 fill

        right_scroll.setWidget(right_inner)
        right_layout.addWidget(right_scroll)
        splitter.addWidget(right_widget)

        splitter.setSizes([900, 400])

        # ── Bottom: batch results tab (resizable via splitter) ──
        self.batch_tab = QTabWidget()
        self.batch_table = BatchResultTable()
        self.batch_table.cellClicked.connect(self._on_table_row_clicked)
        self.batch_tab.addTab(self.batch_table, "批量分析结果")
        self.batch_tab.setMinimumHeight(80)
        self.batch_tab.hide()

        # Vertical splitter: image + control panel on top, batch table below
        vsplitter = QSplitter(Qt.Vertical)
        upper_widget = QWidget()
        upper_widget.setLayout(main_layout)
        vsplitter.addWidget(upper_widget)
        vsplitter.addWidget(self.batch_tab)
        vsplitter.setSizes([600, 250])

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(vsplitter)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        outer_widget = QWidget()
        outer_widget.setLayout(outer_layout)
        self.setCentralWidget(outer_widget)

        # ── Status bar ──
        self.statusBar().showMessage("就绪 - 请打开一张IHC染色图像开始分析")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self._on_threshold_changed()

    # ─── File operations ────────────────────────────────────────────
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开IHC图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.svs);;所有文件 (*)"
        )
        if path:
            self._load_image(path)

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
        self.current_index = 0
        self._load_image(paths[0])
        self._update_nav_label()
        self.statusBar().showMessage(
            f"从文件夹加载 {len(paths)} 张图像, 用 ◀▶ 切换, 点击[批量分析]开始")

    @staticmethod
    def _imread_unicode(path):
        """Defensive image loader: prefer Pillow (better TIFF support), fall back to cv2."""
        img = None
        try:
            pil_img = Image.open(path)
            pil_img.load()
            pil_img = pil_img.convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            try:
                data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                img = None
        return img

    def _load_image(self, path):
        img = self._imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法打开图像:\n{path}")
            return

        self.original_image = img
        self.rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.current_file = path

        self.canvas_original.set_image(self.rgb_image, is_rgb=True)
        self.canvas_original.clear_roi()

        # Auto-run preprocessing + HSV detection
        self._perform_deconvolution()

        self.setWindowTitle(f"FAST IHC Analyzer - {os.path.basename(path)}")
        self.statusBar().showMessage(f"已加载: {path}  |  尺寸: {img.shape[1]}×{img.shape[0]}")

    # ─── Preprocessing + HSV positive detection ────────────────────
    def _perform_deconvolution(self):
        """Preprocessing + HSV positive region detection (follows tiff/ihc_gui.py IHCAnalyzer logic)."""
        if self.rgb_image is None:
            return

        # Step 1: Preprocessing (GaussianBlur + CLAHE on LAB L-channel)
        self.preprocessed_image = self._preprocess_rgb(self.rgb_image)

        # Step 2: HSV positive region detection (delegates to static method)
        mask, self.masked_image, self.positive_ratio = self._detect_positive_hsv(
            self.preprocessed_image, self.hsv_params)
        self.hsv_mask = mask

        # Generate grayscale channels for analysis
        self.dab_channel = cv2.cvtColor(self.masked_image, cv2.COLOR_RGB2GRAY)
        self.hem_channel = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_RGB2GRAY)

        # Display channel images
        # DAB channel: show positive regions (color, non-positive is black)
        self.canvas_dab.set_image(self.masked_image, is_rgb=True)
        # Hematoxylin channel: show preprocessed image
        self.canvas_hem.set_image(self.preprocessed_image, is_rgb=True)

        # Show detection results in status bar
        pos_count = cv2.countNonZero(mask)
        self.statusBar().showMessage(
            f"HSV检测: 阳性像素 {pos_count:,} / {mask.size:,} "
            f"({self.positive_ratio * 100:.1f}%)"
        )

        # Update histogram
        self._update_histogram()

    @staticmethod
    def _preprocess_rgb(rgb):
        """Apply GaussianBlur + CLAHE preprocessing to an RGB image (used for batch analysis)."""
        blurred = cv2.GaussianBlur(rgb, (3, 3), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2RGB)

    @staticmethod
    def _detect_positive_hsv(preprocessed_rgb, params):
        """HSV positive region detection; returns (mask, masked_image, positive_ratio)."""
        hsv = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([params['hue_low'], params['saturation_low'], params['value_low']])
        upper = np.array([params['hue_high'], 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masked_image = cv2.bitwise_and(preprocessed_rgb, preprocessed_rgb, mask=mask)
        total_pixels = mask.size
        positive_pixels = cv2.countNonZero(mask)
        positive_ratio = positive_pixels / total_pixels if total_pixels else 0
        return mask, masked_image, positive_ratio

    def _get_threshold_values(self):
        """Get and normalize current grayscale thresholds."""
        t_high = self.slider_strong.value()
        t_pos = self.slider_moderate.value()
        t_low = self.slider_weak.value()
        t_tissue = self.slider_tissue.value()
        return t_high, t_pos, t_low, t_tissue

    def _update_threshold_info(self):
        """Update threshold info text (HSV detection + grayscale intensity grading)."""
        t_high, t_pos, t_low, t_tissue = self._get_threshold_values()
        if self.lang is self.LANG_ZH:
            self.threshold_info_label.setText(
                "HSV检测阳性区域, 灰度值越高 = 染色越强\n"
                f"强阳性(>={t_high}) | 阳性({t_pos}-{t_high - 1}) | "
                f"弱阳性({t_low}-{t_pos - 1}) | 阴性(<{t_low})"
            )
        else:
            self.threshold_info_label.setText(
                "HSV detects positive regions, higher gray = stronger staining\n"
                f"High+(>={t_high}) | Pos({t_pos}-{t_high - 1}) | "
                f"Low+({t_low}-{t_pos - 1}) | Neg(<{t_low})"
            )

    # ─── Analysis ─────────────────────────────────────────────────
    def analyze_current(self):
        if self.dab_channel is None:
            QMessageBox.warning(self, "提示", "请先打开一张IHC染色图像")
            return
        results = self._calculate_scores()
        self._display_results(results)
        self._create_score_overlay(results)

    def _calculate_scores(self, dab=None, roi=None, positive_ratio=None, thresholds=None):
        """Calculate IHC scores (mirrors tiff/ihc_gui.py IHCAnalyzer logic).
        - Grayscale grading on masked image, thresholds controlled by sliders
        - Higher grayscale = stronger staining within positive regions
        - thresholds: (t_high, t_pos, t_low) three grayscale thresholds
        """
        if dab is None:
            dab = self.dab_channel
        if positive_ratio is None:
            positive_ratio = self.positive_ratio

        # Read thresholds: prefer passed values, otherwise read from sliders
        if thresholds:
            t_high, t_pos, t_low = thresholds
        else:
            t_high = self.slider_strong.value()    # High positive threshold (>=)
            t_pos = self.slider_moderate.value()    # Positive threshold (>=)
            t_low = self.slider_weak.value()        # Low positive threshold (>=)

        # Get analysis region
        if roi is None:
            roi = self.canvas_original.get_roi()

        if roi and not roi.isNull():
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            x = max(0, x)
            y = max(0, y)
            w = min(w, dab.shape[1] - x)
            h = min(h, dab.shape[0] - y)
            gray = dab[y:y+h, x:x+w]
            area_info = f"ROI({x},{y},{w}x{h})"
        else:
            gray = dab
            is_en = hasattr(self, 'lang') and self.lang is self.LANG_EN
            area_info = "Full Image" if is_en else "全图"

        total_pixels = int(gray.size)

        if total_pixels == 0:
            return {
                'negative': 100.0, 'low_pos': 0.0, 'positive': 0.0, 'high_pos': 0.0,
                'h_score': 0.0, 'positive_rate': 0.0,
                'intensity_score': 0,
                'intensity_label': 'Negative' if (hasattr(self, 'lang') and self.lang is self.LANG_EN) else '阴性',
                'intensity_basis': 'No pixels' if (hasattr(self, 'lang') and self.lang is self.LANG_EN) else '无像素',
                'mean_positive_gray': None,
                'proportion_score': 1, 'ihc_score': 0,
                'clinical': 'Negative',
                'clinical_detail': 'Negative [-]' if (hasattr(self, 'lang') and self.lang is self.LANG_EN) else '阴性 [-]',
                'total_pixels': 0, 'tissue_pixels': 0,
                'background_pixels': 0,
                'area_info': area_info
            }

        # ── Intensity grading (using slider thresholds) ──
        # In masked image: excluded pixels = 0 (black), positive pixels retain original grayscale
        n_high = int(np.sum(gray >= t_high))
        n_pos  = int(np.sum((gray >= t_pos) & (gray < t_high)))
        n_low  = int(np.sum((gray >= t_low) & (gray < t_pos)))
        n_neg  = int(np.sum(gray < t_low))

        pct_high = n_high / total_pixels * 100
        pct_pos  = n_pos / total_pixels * 100
        pct_low  = n_low / total_pixels * 100
        pct_neg  = n_neg / total_pixels * 100

        total_pos = pct_high + pct_pos + pct_low
        score_label = 'Positive' if total_pos > 5 else 'Negative'

        # H-Score (kept for compatibility)
        h_score = 1 * pct_low + 2 * pct_pos + 3 * pct_high

        # ── Clinical scoring (matches tiff IHCAnalyzer.calculate_clinical_scores) ──
        # Mean grayscale of positive pixels (gray > 0 excludes masked-out black pixels)
        positive_gray = gray[gray > 0]
        mean_intensity = float(np.mean(positive_gray)) if positive_gray.size else 0

        # Staining intensity score (0-3), using the same thresholds
        is_en = hasattr(self, 'lang') and self.lang is self.LANG_EN
        if mean_intensity < t_low:
            intensity_score = 0
            intensity_label = 'Negative' if is_en else '阴性'
        elif mean_intensity < t_pos:
            intensity_score = 1
            intensity_label = 'Low Positive' if is_en else '弱阳性'
        elif mean_intensity < t_high:
            intensity_score = 2
            intensity_label = 'Positive' if is_en else '阳性'
        else:
            intensity_score = 3
            intensity_label = 'Strong Positive' if is_en else '强阳性'

        intensity_basis = (f"Mean positive gray: {mean_intensity:.1f}"
                           if is_en else f"阳性区域平均灰度: {mean_intensity:.1f}")

        # Positive proportion score (1-4), based on HSV-detected positive_ratio
        pos_pct = positive_ratio * 100
        if pos_pct <= 25:
            proportion_score = 1
        elif pos_pct <= 50:
            proportion_score = 2
        elif pos_pct <= 75:
            proportion_score = 3
        else:
            proportion_score = 4

        # IHC score = intensity x proportion (0-12)
        ihc_score = intensity_score * proportion_score

        # Clinical determination
        if score_label == 'Negative':
            clinical = 'Negative'
            clinical_detail = 'Negative [-]' if is_en else '阴性 [-]'
        elif ihc_score <= 3:
            clinical = 'Positive'
            clinical_detail = 'Low Positive [+]' if is_en else '弱阳性 [+]'
        elif ihc_score <= 6:
            clinical = 'Positive'
            clinical_detail = 'Positive [++]' if is_en else '阳性 [++]'
        else:
            clinical = 'Positive'
            clinical_detail = 'Strong Positive [+++]' if is_en else '强阳性 [+++]'

        # Tissue/background pixel statistics
        tissue_pixels = int(np.sum(gray > 0))
        background_pixels = int(np.sum(gray == 0))

        # Grading masks (for overlay display, using the same thresholds)
        high_mask = (gray >= t_high)
        pos_mask  = (gray >= t_pos) & (gray < t_high)
        low_mask  = (gray >= t_low) & (gray < t_pos)
        neg_mask  = (gray < t_low)

        return {
            'negative': pct_neg, 'low_pos': pct_low,
            'positive': pct_pos, 'high_pos': pct_high,
            'h_score': h_score, 'positive_rate': total_pos,
            'intensity_score': intensity_score,
            'intensity_label': intensity_label,
            'intensity_basis': intensity_basis,
            'mean_positive_gray': mean_intensity if positive_gray.size else None,
            'proportion_score': proportion_score,
            'ihc_score': ihc_score,
            'clinical': clinical,
            'clinical_detail': clinical_detail,
            'total_pixels': total_pixels,
            'tissue_pixels': tissue_pixels,
            'background_pixels': background_pixels,
            'area_info': area_info,
            'masks': {
                'negative': neg_mask, 'low_pos': low_mask,
                'positive': pos_mask, 'high_pos': high_mask,
            }
        }

    def _display_results(self, results):
        """Display scoring results (bilingual)."""
        t_high, t_pos, t_low, _t_tissue = self._get_threshold_values()
        is_en = self.lang is self.LANG_EN

        if is_en:
            text = f"""{'='*42}
  IHC Scoring Report
{'='*42}
  File: {os.path.basename(self.current_file)}
  Region: {results['area_info']}
  Total Pixels: {results['total_pixels']:,} px
  Tissue (HSV+): {results['tissue_pixels']:,} px
  Background: {results['background_pixels']:,} px
{'_'*42}
  High Positive (>={t_high}): {results['high_pos']:6.1f}%
  Positive ({t_pos}-{t_high - 1}): {results['positive']:6.1f}%
  Low Positive ({t_low}-{t_pos - 1}): {results['low_pos']:6.1f}%
  Negative (<{t_low}): {results['negative']:6.1f}%
{'_'*42}
  Intensity: {results['intensity_score']}  [{results['intensity_label']}]
  Proportion: {results['proportion_score']}  (0-4)
  Basis: {results['intensity_basis']}
{'_'*42}
  H-Score: {results['h_score']:6.1f} / 300
  Positive Rate: {results['positive_rate']:6.1f}%
  IHC Score: {results['ihc_score']:>2d}  (0-12)
{'='*42}
  Result: {results['clinical_detail']}
{'='*42}"""
        else:
            text = f"""{'='*42}
  IHC 评分报告
{'='*42}
  文件: {os.path.basename(self.current_file)}
  区域: {results['area_info']}
  总像素: {results['total_pixels']:,} px
  组织像素(HSV阳性): {results['tissue_pixels']:,} px
  背景像素(非阳性): {results['background_pixels']:,} px
{'_'*42}
  强阳性 (>={t_high}): {results['high_pos']:6.1f}%
  阳性 ({t_pos}-{t_high - 1}): {results['positive']:6.1f}%
  弱阳性 ({t_low}-{t_pos - 1}): {results['low_pos']:6.1f}%
  阴性 (<{t_low}): {results['negative']:6.1f}%
{'_'*42}
  强度评分: {results['intensity_score']}  [{results['intensity_label']}]
  比例评分: {results['proportion_score']}  (0-4)
  规则说明: {results['intensity_basis']}
{'_'*42}
  H-Score: {results['h_score']:6.1f} / 300
  阳性率: {results['positive_rate']:6.1f}%
  IHC评分: {results['ihc_score']:>2d}  (0-12)
{'='*42}
  判定: {results['clinical_detail']}
{'='*42}"""

        self.result_text.setPlainText(text)

        # Update pie chart
        pie_lang = 'zh' if self.lang is self.LANG_ZH else 'en'
        self.pie_chart.plot_scores(
            results['negative'], results['low_pos'],
            results['positive'], results['high_pos'], lang=pie_lang
        )

        self.statusBar().showMessage(
            f"IHC评分={results['ihc_score']} | "
            f"{results['clinical_detail']} | "
            f"H-Score={results['h_score']:.1f}"
        )

    def _create_score_overlay(self, results):
        """Create score overlay image.
        Only colorize HSV-detected positive regions by intensity; non-positive
        regions are shown dimmed from the original image.
        """
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
            hsv_mask_roi = self.hsv_mask[y:y+h, x:x+w] if self.hsv_mask is not None else None
        else:
            overlay = self.rgb_image.copy()
            hsv_mask_roi = self.hsv_mask

        # Colorize positive regions by intensity grade
        alpha = 0.45

        # Low Positive - green (within HSV positive regions only)
        overlay[masks['low_pos']] = (
            overlay[masks['low_pos']] * (1 - alpha) +
            np.array([102, 187, 106]) * alpha
        ).astype(np.uint8)

        # Positive - orange
        overlay[masks['positive']] = (
            overlay[masks['positive']] * (1 - alpha) +
            np.array([255, 167, 38]) * alpha
        ).astype(np.uint8)

        # High Positive - red
        overlay[masks['high_pos']] = (
            overlay[masks['high_pos']] * (1 - alpha) +
            np.array([239, 83, 80]) * alpha
        ).astype(np.uint8)

        # Non-positive regions (not detected by HSV): dim slightly to highlight positive areas
        if hsv_mask_roi is not None:
            non_positive = (hsv_mask_roi == 0)
            overlay[non_positive] = (overlay[non_positive] * 0.7).astype(np.uint8)

        self.score_mask = overlay
        self.canvas_score.set_image(overlay, is_rgb=True)
        self.image_tabs.setCurrentIndex(3)  # Switch to score overlay tab

    def _update_histogram(self):
        """Update grayscale histogram."""
        if self.dab_channel is None:
            return
        roi = self.canvas_original.get_roi()
        if roi and not roi.isNull():
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            x, y = max(0, x), max(0, y)
            data = self.dab_channel[y:y+h, x:x+w]
        else:
            data = self.dab_channel

        # Grayscale threshold lines: High+, Positive, Low+
        thresholds = [
            self.slider_strong.value(),   # High+ boundary
            self.slider_moderate.value(),  # Positive boundary
            self.slider_weak.value(),      # Low+ boundary
        ]
        self.histogram.plot_histogram(
            data, "灰度分布",
            thresholds, colors=['#ef5350', '#ffa726', '#66bb6a']
        )

    # ─── Threshold changes ─────────────────────────────────────────
    def _on_threshold_changed(self):
        # Ensure descending threshold order: high+ > positive > low+
        s = max(self.slider_strong.value(), 2)   # High positive >= (highest)
        m = self.slider_moderate.value()          # Positive >=
        w = self.slider_weak.value()              # Low positive >= (lowest)

        # Enforce strong > moderate > weak >= 1
        if m >= s:
            m = s - 1
        if m < 1:
            m = 1
        if w >= m:
            w = m - 1
        if w < 0:
            w = 0

        if s != self.slider_strong.value():
            self.slider_strong.blockSignals(True)
            self.slider_strong.setValue(s)
            self.slider_strong.blockSignals(False)
        if m != self.slider_moderate.value():
            self.slider_moderate.blockSignals(True)
            self.slider_moderate.setValue(m)
            self.slider_moderate.blockSignals(False)
        if w != self.slider_weak.value():
            self.slider_weak.blockSignals(True)
            self.slider_weak.setValue(w)
            self.slider_weak.blockSignals(False)

        self.lbl_strong.setText(str(self.slider_strong.value()))
        self.lbl_moderate.setText(str(self.slider_moderate.value()))
        self.lbl_weak.setText(str(self.slider_weak.value()))
        self.lbl_tissue.setText(str(self.slider_tissue.value()))

        self._update_threshold_info()
        self._update_histogram()

    def _set_thresholds(self, high, pos, low):
        # Block signals, set all three, then trigger validation once
        self.slider_strong.blockSignals(True)
        self.slider_moderate.blockSignals(True)
        self.slider_weak.blockSignals(True)
        self.slider_strong.setValue(high)
        self.slider_moderate.setValue(pos)
        self.slider_weak.setValue(low)
        self.slider_strong.blockSignals(False)
        self.slider_moderate.blockSignals(False)
        self.slider_weak.blockSignals(False)
        self._on_threshold_changed()

    # ─── Language toggle ────────────────────────────────────────────
    def _toggle_language(self):
        if self.lang is self.LANG_ZH:
            self.lang = self.LANG_EN
        else:
            self.lang = self.LANG_ZH
        self._apply_language()

    def _apply_language(self):
        L = self.lang
        self.setWindowTitle(L['title'])
        self.btn_lang.setText("🌐 " + L['lang_switch'])

        # Toolbar buttons
        self.btn_open.setText("📂 " + L['toolbar_open'])
        self.btn_folder.setText("📂 " + L['toolbar_folder'])
        self.btn_prev.setText("◀ " + L['toolbar_prev'])
        self.btn_next.setText(L['toolbar_next'] + " ▶")
        self.btn_roi.setText("[+] " + L['toolbar_roi'])
        self.btn_clear_roi.setText("[x] " + L['toolbar_clear_roi'])
        self.btn_analyze.setText("▶ " + L['toolbar_analyze'])
        self.btn_batch_analyze.setText("▶▶ " + L['toolbar_batch_analyze'])
        self.btn_export.setText("💾 " + L['toolbar_export'])
        self.btn_save_img.setText("🖼 " + L['toolbar_save'])

        # Menu items
        self.act_open.setText(L['open'])
        self.act_folder.setText(L['open_folder'])
        self.act_export.setText(L['export'])
        self.act_save_img.setText(L['save_img'])

        # Right panel - GroupBox titles
        self.grp_deconv.setTitle(L['grp_deconv'])
        self.lbl_detect_info.setText(L['detect_info'])
        self.grp_thresh.setTitle(L['grp_thresh'])
        self.grp_hist.setTitle(L['grp_hist'])
        self.grp_result.setTitle(L['grp_result'])

        # Right panel - labels
        self.lbl_stain.setText(L['stain_label'])
        self.chk_auto_balance.setText(L['auto_wb'])
        self.lbl_high_tag.setText(L['lbl_high'])
        self.lbl_pos_tag.setText(L['lbl_pos'])
        self.lbl_low_tag.setText(L['lbl_low'])
        self.lbl_preset.setText(L['preset_label'])
        self.btn_preset_default.setText(L['preset_default'])
        self.btn_preset_std.setText(L['preset_std'])
        self.btn_preset_strict.setText(L['preset_strict'])
        self.btn_preset_loose.setText(L['preset_loose'])
        self.lbl_bg_tag.setText(L['bg_label'])

        # Image tabs
        self.image_tabs.setTabText(0, L['tab_original'])
        self.image_tabs.setTabText(1, L['tab_dab'])
        self.image_tabs.setTabText(2, L['tab_hem'])
        self.image_tabs.setTabText(3, L['tab_score'])

        # Batch table
        self.batch_tab.setTabText(0, L['batch_tab'])
        for col, header in enumerate(L['table_headers']):
            self.batch_table.setHorizontalHeaderItem(col, QTableWidgetItem(header))

        # Refresh threshold info text
        self._update_threshold_info()

        self.statusBar().showMessage(L['status_ready'])

        # After language switch, recalculate and redisplay results if available
        if self.dab_channel is not None:
            results = self._calculate_scores()
            self._display_results(results)

    # ─── Image navigation ──────────────────────────────────────────
    def _prev_image(self):
        if not self.batch_files or self.current_index <= 0:
            return
        self.current_index -= 1
        self._navigate_to(self.current_index)

    def _next_image(self):
        if not self.batch_files or self.current_index >= len(self.batch_files) - 1:
            return
        self.current_index += 1
        self._navigate_to(self.current_index)

    def _navigate_to(self, index):
        """Navigate to the image at the given index and restore cached analysis results."""
        path = self.batch_files[index]
        self._update_nav_label()

        # If cached image data exists, restore directly (avoid reloading and reprocessing)
        if index in self.batch_image_cache:
            rgb, preprocessed, masked_img, mask, dab, pos_ratio = self.batch_image_cache[index]
            self.rgb_image = rgb
            self.original_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.preprocessed_image = preprocessed
            self.masked_image = masked_img
            self.hsv_mask = mask
            self.dab_channel = dab
            self.hem_channel = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
            self.positive_ratio = pos_ratio
            self.current_file = path

            # Refresh all image panels
            self.canvas_original.set_image(rgb, is_rgb=True)
            self.canvas_dab.set_image(masked_img, is_rgb=True)
            self.canvas_hem.set_image(preprocessed, is_rgb=True)
            self._update_histogram()

            self.setWindowTitle(f"FAST IHC Analyzer - {os.path.basename(path)}")
        else:
            # No cache, load normally
            self._load_image(path)

        # If cached analysis results exist, auto-display scores
        if index in self.batch_results_cache:
            results = self.batch_results_cache[index]
            self._display_results(results)
            self._create_score_overlay(results)

        # Highlight corresponding row in the table
        if self.batch_table.rowCount() > index:
            self.batch_table.selectRow(index)
            self.batch_table.scrollToItem(
                self.batch_table.item(index, 0))

    def _on_table_row_clicked(self, row, _col):
        """Switch to the corresponding image when a table row is clicked."""
        if 0 <= row < len(self.batch_files):
            self.current_index = row
            self._navigate_to(row)

    def _update_nav_label(self):
        if self.batch_files:
            self.lbl_image_index.setText(
                f"{self.current_index + 1}/{len(self.batch_files)}")
        else:
            self.lbl_image_index.setText("")

    # ─── ROI ─────────────────────────────────────────────────────
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

    # ─── Batch analysis ────────────────────────────────────────────
    def batch_analyze(self):
        if not self.batch_files:
            QMessageBox.information(self, "提示", "请先通过[打开文件夹]加载图像")
            return

        self.batch_tab.show()
        self.batch_table.setRowCount(0)
        self.batch_results_cache.clear()
        self.batch_image_cache.clear()

        self.progress_bar.show()
        self.progress_bar.setRange(0, len(self.batch_files))

        thresholds = (self.slider_strong.value(),
                      self.slider_moderate.value(),
                      self.slider_weak.value())

        for i, path in enumerate(self.batch_files):
            self.progress_bar.setValue(i)
            QApplication.processEvents()

            img = self._imread_unicode(path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocessing + HSV detection
            preprocessed = self._preprocess_rgb(rgb)
            mask, masked_img, pos_ratio = self._detect_positive_hsv(
                preprocessed, self.hsv_params)
            dab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

            results = self._calculate_scores(
                dab=dab, roi=None, positive_ratio=pos_ratio,
                thresholds=thresholds)

            results['filename'] = os.path.basename(path)

            # Cache results and image data (for navigation)
            self.batch_results_cache[i] = results
            self.batch_image_cache[i] = (rgb, preprocessed, masked_img, mask, dab, pos_ratio)

            self.batch_table.add_result(results)

        self.progress_bar.setValue(len(self.batch_files))
        self.progress_bar.hide()

        n = len(self.batch_results_cache)
        self.statusBar().showMessage(f"批量分析完成: {n} 张图像")

        # Auto-navigate to the first image and display its analysis results
        if self.batch_files:
            self.current_index = 0
            self._navigate_to(0)

    # ─── Export ───────────────────────────────────────────────────
    def _csv_headers(self):
        """Return CSV headers based on the current language."""
        if self.lang is self.LANG_EN:
            return ['Filename', 'Total Pixels', 'High+(%)', 'Pos(%)', 'Low+(%)',
                    'Neg(%)', 'Clinical', 'Intensity', 'Proportion', 'IHC Score']
        return ['图片名称', '总像素', '高强阳(%)', '中阳(%)', '低阳(%)',
                '阴性(%)', '临床判定', '强度评分', '比例评分', 'IHC评分']

    def export_results(self):
        is_en = self.lang is self.LANG_EN
        if self.batch_table.rowCount() == 0:
            # Single image result only
            if self.dab_channel is None:
                msg = "No results to export" if is_en else "没有可导出的结果"
                QMessageBox.information(self, "Info" if is_en else "提示", msg)
                return
            results = self._calculate_scores()
            dlg_title = "Export Results" if is_en else "导出结果"
            dlg_filter = "CSV Files (*.csv)" if is_en else "CSV文件 (*.csv)"
            path, _ = QFileDialog.getSaveFileName(
                self, dlg_title,
                f"ihc_result_{datetime.now():%Y%m%d_%H%M%S}.csv", dlg_filter)
            if path:
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(self._csv_headers())
                    writer.writerow([
                        os.path.basename(self.current_file),
                        results['total_pixels'],
                        f"{results['high_pos']:.2f}",
                        f"{results['positive']:.2f}",
                        f"{results['low_pos']:.2f}",
                        f"{results['negative']:.2f}",
                        results['clinical'],
                        results['intensity_score'],
                        results['proportion_score'],
                        results['ihc_score'],
                    ])
                msg = f"Exported: {path}" if is_en else f"结果已导出: {path}"
                self.statusBar().showMessage(msg)
        else:
            # Batch results
            dlg_title = "Export Batch Results" if is_en else "导出批量结果"
            dlg_filter = "CSV Files (*.csv)" if is_en else "CSV文件 (*.csv)"
            path, _ = QFileDialog.getSaveFileName(
                self, dlg_title,
                f"ihc_batch_{datetime.now():%Y%m%d_%H%M%S}.csv", dlg_filter)
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
                msg = f"Exported: {path}" if is_en else f"批量结果已导出: {path}"
                self.statusBar().showMessage(msg)

    def save_analysis_image(self):
        is_en = self.lang is self.LANG_EN
        if self.score_mask is None:
            msg = "Please analyze first" if is_en else "请先执行分析"
            QMessageBox.information(self, "Info" if is_en else "提示", msg)
            return
        dlg_title = "Save Analysis Image" if is_en else "保存分析图像"
        path, _ = QFileDialog.getSaveFileName(
            self, dlg_title,
            f"ihc_analysis_{datetime.now():%Y%m%d_%H%M%S}.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)"
        )
        if path:
            save_img = cv2.cvtColor(self.score_mask, cv2.COLOR_RGB2BGR)
            ext = os.path.splitext(path)[1]
            result, buf = cv2.imencode(ext, save_img)
            if result:
                buf.tofile(path)
            msg = f"Image saved: {path}" if is_en else f"分析图像已保存: {path}"
            self.statusBar().showMessage(msg)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FAST IHC Analyzer")

    # High DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Set app-level icon (taskbar / dock)
    for name in ('icon.ico', 'icon.png'):
        p = IHCScorer._get_resource_path(name)
        if os.path.exists(p):
            app.setWindowIcon(QIcon(p))
            break

    window = IHCScorer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
