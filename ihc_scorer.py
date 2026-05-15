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
    """Zoomable, pannable image display widget with freehand ROI selection."""
    roi_selected = pyqtSignal(object)  # emits list of QPoint (polygon)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background-color: #10131c; border: 1px solid rgba(100, 140, 220, 0.06); "
            "border-radius: 10px; color: #3a4a68; font-size: 18px;"
        )

        self._pixmap = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._dragging = False
        self._drag_start = QPoint()
        self._selecting_roi = False
        self._roi_mode = False
        self._roi_points = []       # freehand polygon points (image coords)
        self._drawing_points = []   # points being drawn (image coords)
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

    def set_empty_hint(self, text):
        """Set the placeholder text shown when no image is loaded."""
        self._empty_hint = text

    def _update_display(self):
        if self._pixmap is None:
            hint = getattr(self, '_empty_hint', '')
            self.setText(hint)
            return
        scaled = self._pixmap.scaled(
            int(self._pixmap.width() * self._scale),
            int(self._pixmap.height() * self._scale),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        # Draw freehand ROI overlay
        points_to_draw = self._drawing_points if self._selecting_roi else self._roi_points
        if len(points_to_draw) >= 2:
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.Antialiasing)
            # Semi-transparent fill
            if not self._selecting_roi and len(points_to_draw) >= 3:
                from PyQt5.QtGui import QPolygonF, QBrush
                from PyQt5.QtCore import QPointF
                poly = QPolygonF([QPointF(p.x() * self._scale, p.y() * self._scale) for p in points_to_draw])
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 220, 50, 30)))
                painter.drawPolygon(poly)
            # Outline
            pen = QPen(QColor(255, 220, 50), 2, Qt.SolidLine if self._selecting_roi else Qt.DashLine)
            painter.setPen(pen)
            for i in range(len(points_to_draw) - 1):
                p1 = points_to_draw[i]
                p2 = points_to_draw[i + 1]
                painter.drawLine(
                    int(p1.x() * self._scale), int(p1.y() * self._scale),
                    int(p2.x() * self._scale), int(p2.y() * self._scale),
                )
            # Close the polygon if finished
            if not self._selecting_roi and len(points_to_draw) >= 3:
                p1 = points_to_draw[-1]
                p2 = points_to_draw[0]
                painter.drawLine(
                    int(p1.x() * self._scale), int(p1.y() * self._scale),
                    int(p2.x() * self._scale), int(p2.y() * self._scale),
                )
            painter.end()
        self.setPixmap(scaled)

    def set_roi_mode(self, enabled):
        self._roi_mode = enabled
        self.setCursor(QCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor))

    def clear_roi(self):
        self._roi_points = []
        self._drawing_points = []
        self._update_display()

    def get_roi(self):
        """Return bounding QRect of the freehand polygon, or None."""
        if len(self._roi_points) < 3:
            return None
        xs = [p.x() for p in self._roi_points]
        ys = [p.y() for p in self._roi_points]
        return QRect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def get_roi_mask(self, shape):
        """Return a binary numpy mask (h, w) for the freehand polygon ROI.
        Returns None if no ROI is set."""
        if len(self._roi_points) < 3:
            return None
        h, w = shape[:2]
        pts = np.array([[p.x(), p.y()] for p in self._roi_points], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

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
                self._drawing_points = [self._widget_to_image(event.pos())]
            else:
                self._dragging = True
                self._drag_start = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._selecting_roi:
            pt = self._widget_to_image(event.pos())
            # Only add point if moved enough (avoid too many points)
            if self._drawing_points:
                last = self._drawing_points[-1]
                if abs(pt.x() - last.x()) > 2 or abs(pt.y() - last.y()) > 2:
                    self._drawing_points.append(pt)
            else:
                self._drawing_points.append(pt)
            self._update_display()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self._selecting_roi:
                self._selecting_roi = False
                if len(self._drawing_points) >= 10:
                    self._roi_points = list(self._drawing_points)
                    self.roi_selected.emit(self._roi_points)
                self._drawing_points = []
                self._update_display()
            self._dragging = False

    def _widget_to_image(self, pos):
        """Convert widget coordinates to image coordinates."""
        if self._pixmap is None:
            return QPoint(0, 0)
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
        self.fig.patch.set_facecolor('#1a1e2e')
        super().__init__(self.fig)
        self.setMinimumHeight(180)

    def plot_histogram(self, data, title="", thresholds=None, colors=None):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#131620')
        ax.tick_params(colors='#4a5570', labelsize=8)
        ax.set_title(title, color='#7888a8', fontsize=10, fontweight='medium')

        if data is not None and len(data) > 0:
            ax.hist(data.ravel(), bins=256, range=(0, 255),
                    color='#3088ff', alpha=0.45, edgecolor='none')

            if thresholds:
                color_list = colors or ['#66bb6a', '#ffa726', '#ef5350']
                for i, t in enumerate(thresholds):
                    c = color_list[i] if i < len(color_list) else '#ffffff'
                    ax.axvline(x=t, color=c, linestyle='--', linewidth=1.5)

        ax.set_xlim(0, 255)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.label.set_color('#4a5570')
        ax.yaxis.label.set_color('#4a5570')
        self.fig.tight_layout()
        self.draw()


class ScorePieChart(FigureCanvas):
    """Pie chart for score distribution display."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3, 3), dpi=80)
        self.fig.patch.set_facecolor('#1a1e2e')
        super().__init__(self.fig)
        self.setMinimumHeight(200)

    def plot_scores(self, negative, low_pos, positive, high_pos, lang='zh'):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#1a1e2e')

        values = [negative, low_pos, positive, high_pos]
        if lang == 'zh':
            labels = [f'阴性\n{negative:.1f}%', f'弱阳性\n{low_pos:.1f}%',
                      f'阳性\n{positive:.1f}%', f'强阳性\n{high_pos:.1f}%']
            title = '评分分布'
        else:
            labels = [f'Neg\n{negative:.1f}%', f'Low+\n{low_pos:.1f}%',
                      f'Pos\n{positive:.1f}%', f'High+\n{high_pos:.1f}%']
            title = 'Score Distribution'
        colors_list = ['#3080e0', '#20a060', '#d09020', '#d04040']

        non_zero = [(v, l, c) for v, l, c in zip(values, labels, colors_list) if v > 0.1]
        if non_zero:
            vals, labs, cols = zip(*non_zero)
            wedges, texts = ax.pie(vals, labels=labs, colors=cols,
                                    startangle=90,
                                    wedgeprops={'linewidth': 2, 'edgecolor': '#1a1e2e'},
                                    textprops={'color': '#7888a8', 'fontsize': 9})
        ax.set_title(title, color='#7888a8', fontsize=11, fontweight='medium')
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
                background-color: #131620;
                color: #a0aac0;
                gridline-color: #1a1e2e;
                alternate-background-color: #161a28;
                border: none;
                font-size: 15px;
            }
            QTableWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #1a1e2e;
            }
            QTableWidget::item:selected {
                background-color: rgba(48, 136, 255, 0.15);
                color: #6eb4ff;
            }
            QHeaderView::section {
                background-color: #131620;
                color: #5068a0;
                padding: 8px 10px;
                border: none;
                border-bottom: 1px solid #252a3c;
                font-weight: 600;
                font-size: 14px;
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
                    item.setForeground(QColor('#d09020'))
                else:
                    item.setForeground(QColor('#3080e0'))
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
        'empty_hint': '请打开图像或文件夹开始分析',
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
        'empty_hint': 'Open an image or folder to start analysis',
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
            'hue_low': 0, 'hue_high': 30,
            'saturation_low': 20, 'value_low': 30
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
        # ── Neo-Lab: layered midnight-blue ──
        self.setStyleSheet("""
            /* ═══ Base — midnight navy, NOT black ═══ */
            QMainWindow {
                background-color: #131620;
            }
            QWidget {
                color: #c0c8d8;
                font-size: 16px;
                font-family: "PingFang SC", "Helvetica Neue", "Microsoft YaHei", "Segoe UI", sans-serif;
            }

            /* ═══ GroupBox — elevated card ═══ */
            QGroupBox {
                border: 1px solid rgba(100, 140, 220, 0.1);
                margin-top: 14px;
                padding: 24px 14px 14px 14px;
                font-weight: 600;
                font-size: 16px;
                color: #6eb4ff;
                background-color: #1a1e2e;
                border-radius: 14px;
            }
            QGroupBox::title {
                subcontrol-position: top left;
                padding: 4px 14px;
                color: #6eb4ff;
            }

            /* ═══ Default Buttons ═══ */
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 8px 18px;
                color: #7080a0;
                min-height: 28px;
                font-weight: 500;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(80, 140, 255, 0.08);
                border: 1px solid rgba(80, 140, 255, 0.18);
                color: #d0d8e8;
            }
            QPushButton:pressed {
                background-color: rgba(80, 140, 255, 0.15);
            }
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2060e0, stop:1 #3088ff);
                border: 1px solid rgba(80, 160, 255, 0.4);
                color: #ffffff;
                font-weight: 600;
                border-radius: 8px;
                padding: 8px 22px;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2870f0, stop:1 #40a0ff);
                border-color: rgba(100, 180, 255, 0.6);
            }
            QPushButton#primaryBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1850c0, stop:1 #2870e0);
            }

            /* ═══ Sliders ═══ */
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background: #1e2236;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.35, stop:0 #80c0ff, stop:0.8 #3080e0, stop:1 #1850a0);
                border: 1px solid rgba(80, 160, 255, 0.5);
                width: 14px;
                height: 14px;
                margin: -6px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1850a0, stop:1 #3088ff);
                border-radius: 2px;
            }

            /* ═══ ComboBox ═══ */
            QComboBox {
                background-color: #1a1e2e;
                border: 1px solid #252a3c;
                border-radius: 8px;
                padding: 6px 12px;
                color: #c0c8d8;
            }
            QComboBox:hover { border-color: #4080d0; }
            QComboBox::drop-down { border: none; padding-right: 10px; }
            QComboBox QAbstractItemView {
                background-color: #1a1e2e;
                color: #c0c8d8;
                selection-background-color: #253060;
                selection-color: #80c0ff;
                border: 1px solid #252a3c;
                border-radius: 8px;
            }

            /* ═══ SpinBox ═══ */
            QSpinBox, QDoubleSpinBox {
                background-color: #1a1e2e;
                border: 1px solid #252a3c;
                border-radius: 8px;
                padding: 4px 10px;
                color: #c0c8d8;
            }
            QSpinBox:hover, QDoubleSpinBox:hover { border-color: #4080d0; }

            /* ═══ Tabs ═══ */
            QTabWidget::pane {
                border: none;
                background: #131620;
            }
            QTabBar { background: transparent; }
            QTabBar::tab {
                background: transparent;
                color: #4a5570;
                padding: 12px 26px;
                border: none;
                border-bottom: 2px solid transparent;
                font-size: 16px;
                font-weight: 500;
                margin-right: 4px;
            }
            QTabBar::tab:hover {
                color: #8090b0;
                border-bottom: 2px solid #252a3c;
            }
            QTabBar::tab:selected {
                color: #6eb4ff;
                border-bottom: 2px solid #3088ff;
            }

            /* ═══ StatusBar ═══ */
            QStatusBar {
                background-color: #0e1118;
                color: #4a5570;
                border-top: 1px solid #1a1e2e;
                padding: 4px 14px;
                font-size: 14px;
            }

            /* ═══ TextEdit ═══ */
            QTextEdit {
                background-color: #161a28;
                color: #a0aac0;
                border: 1px solid rgba(100, 140, 220, 0.08);
                border-radius: 12px;
                padding: 14px;
                selection-background-color: #253060;
                selection-color: #80c0ff;
            }

            /* ═══ MenuBar ═══ */
            QMenuBar {
                background-color: #0e1118;
                color: #7080a0;
                font-size: 16px;
                padding: 4px 8px;
                border: none;
            }
            QMenuBar::item {
                padding: 6px 14px;
                border-radius: 6px;
            }
            QMenuBar::item:selected {
                background-color: #1a1e2e;
                color: #d0d8e8;
            }
            QMenu {
                background-color: #1a1e2e;
                color: #c0c8d8;
                border: 1px solid #252a3c;
                border-radius: 12px;
                padding: 6px;
                font-size: 14px;
            }
            QMenu::item {
                padding: 10px 28px;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background-color: #253060;
                color: #80c0ff;
            }
            QMenu::separator {
                height: 1px;
                background: #252a3c;
                margin: 4px 16px;
            }

            /* ═══ Toolbar ═══ */
            QToolBar {
                background-color: #0e1118;
                border-bottom: 1px solid #1a1e2e;
                spacing: 2px;
                padding: 6px 12px;
            }
            QToolBar::separator {
                width: 0;
                background: transparent;
                margin: 0 4px;
            }

            /* ═══ ProgressBar ═══ */
            QProgressBar {
                border: 1px solid #252a3c;
                border-radius: 4px;
                text-align: center;
                color: #c0c8d8;
                background: #1a1e2e;
                font-size: 11px;
                max-height: 6px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2060e0, stop:1 #40a0ff);
                border-radius: 3px;
            }

            /* ═══ ScrollBar ═══ */
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: transparent;
                width: 5px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #252a3c;
                border-radius: 2px;
                min-height: 40px;
            }
            QScrollBar::handle:vertical:hover { background: #354060; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: transparent;
                height: 5px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #252a3c;
                border-radius: 2px;
                min-width: 40px;
            }
            QScrollBar::handle:horizontal:hover { background: #354060; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

            /* ═══ Splitter ═══ */
            QSplitter::handle { background: transparent; }
            QSplitter::handle:horizontal { width: 6px; }
            QSplitter::handle:vertical { height: 6px; }
            QSplitter::handle:hover { background: #1a1e2e; }
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
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Style constants for button groups — layered midnight-blue
        _grp_style = """
            QWidget#toolGroup {
                background-color: #181c2a;
                border: 1px solid rgba(100, 140, 220, 0.1);
                border-radius: 10px;
            }
        """
        _btn_style_normal = """
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 7px 16px;
                color: #6070a0;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(80, 140, 255, 0.1);
                color: #c0d0f0;
            }
            QPushButton:pressed {
                background-color: rgba(80, 140, 255, 0.18);
            }
        """
        _btn_style_nav = """
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 7px 12px;
                color: #405078;
                font-size: 17px;
                font-weight: 400;
                min-width: 20px;
            }
            QPushButton:hover {
                background-color: rgba(80, 140, 255, 0.1);
                color: #6eb4ff;
            }
            QPushButton:pressed {
                background-color: rgba(80, 140, 255, 0.18);
            }
        """
        _btn_style_primary = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2060e0, stop:1 #3088ff);
                border: 1px solid rgba(80, 160, 255, 0.35);
                border-radius: 7px;
                padding: 7px 22px;
                color: #fff;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2870f0, stop:1 #40a0ff);
                border-color: rgba(100, 180, 255, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1850c0, stop:1 #2870e0);
            }
        """
        _btn_style_accent = """
            QPushButton {
                background: transparent;
                border: 1px solid #252a3c;
                border-radius: 7px;
                padding: 7px 16px;
                color: #6078a0;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                border-color: rgba(80, 140, 255, 0.3);
                color: #b0c0e0;
                background-color: rgba(80, 140, 255, 0.06);
            }
            QPushButton:pressed {
                background-color: rgba(80, 140, 255, 0.12);
            }
        """
        _btn_style_checkable = """
            QPushButton {
                background: transparent;
                border: 1px solid #252a3c;
                border-radius: 7px;
                padding: 7px 16px;
                color: #6070a0;
                font-size: 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                border-color: rgba(80, 140, 255, 0.25);
                color: #b0c0e0;
                background-color: rgba(80, 140, 255, 0.06);
            }
            QPushButton:checked {
                background-color: rgba(48, 136, 255, 0.15);
                border-color: rgba(80, 160, 255, 0.4);
                color: #6eb4ff;
            }
            QPushButton:pressed {
                background-color: rgba(48, 136, 255, 0.22);
            }
        """

        def _make_group(widgets):
            """Wrap buttons in a rounded container."""
            grp = QWidget()
            grp.setObjectName("toolGroup")
            grp.setStyleSheet(_grp_style)
            lay = QHBoxLayout(grp)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(2)
            for w in widgets:
                lay.addWidget(w)
            return grp

        # -- File group (highlighted) --
        _btn_style_file = """
            QPushButton {
                background: transparent;
                border: 1px solid rgba(80, 160, 255, 0.3);
                border-radius: 7px;
                padding: 7px 16px;
                color: #6eb4ff;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(80, 140, 255, 0.12);
                border-color: rgba(80, 160, 255, 0.5);
                color: #a0d0ff;
            }
            QPushButton:pressed {
                background-color: rgba(80, 140, 255, 0.2);
            }
        """
        self.btn_open = QPushButton("打开图像")
        self.btn_open.setStyleSheet(_btn_style_file)
        self.btn_open.clicked.connect(self.open_image)

        self.btn_folder = QPushButton("打开文件夹")
        self.btn_folder.setStyleSheet(_btn_style_file)
        self.btn_folder.clicked.connect(self.open_folder)

        toolbar.addWidget(_make_group([self.btn_open, self.btn_folder]))

        # -- Navigation group --
        self.btn_prev = QPushButton("<")
        self.btn_prev.setStyleSheet(_btn_style_nav)
        self.btn_prev.clicked.connect(self._prev_image)

        self.lbl_image_index = QLabel("")
        self.lbl_image_index.setAlignment(Qt.AlignCenter)
        self.lbl_image_index.setStyleSheet(
            "color: #6eb4ff; font-weight: 700; font-size: 15px; "
            "padding: 0 2px; min-width: 36px; border: none;")

        self.btn_next = QPushButton(">")
        self.btn_next.setStyleSheet(_btn_style_nav)
        self.btn_next.clicked.connect(self._next_image)

        toolbar.addWidget(_make_group([self.btn_prev, self.lbl_image_index, self.btn_next]))

        # -- ROI group --
        self.btn_roi = QPushButton("选择ROI")
        self.btn_roi.setStyleSheet(_btn_style_checkable)
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self._toggle_roi_mode)

        self.btn_clear_roi = QPushButton("清除ROI")
        self.btn_clear_roi.setStyleSheet(_btn_style_normal)
        self.btn_clear_roi.clicked.connect(self._clear_roi)

        toolbar.addWidget(_make_group([self.btn_roi, self.btn_clear_roi]))

        # -- Analysis group --
        self.btn_analyze = QPushButton("分析")
        self.btn_analyze.setObjectName("primaryBtn")
        self.btn_analyze.setStyleSheet(_btn_style_primary)
        self.btn_analyze.clicked.connect(self.analyze_current)

        self.btn_batch_analyze = QPushButton("批量分析")
        self.btn_batch_analyze.setObjectName("primaryBtn")
        self.btn_batch_analyze.setStyleSheet(_btn_style_primary)
        self.btn_batch_analyze.clicked.connect(self.batch_analyze)

        toolbar.addWidget(_make_group([self.btn_analyze, self.btn_batch_analyze]))

        # -- Export group --
        self.btn_export = QPushButton("导出CSV")
        self.btn_export.setStyleSheet(_btn_style_accent)
        self.btn_export.clicked.connect(self.export_results)

        self.btn_save_img = QPushButton("保存图像")
        self.btn_save_img.setStyleSheet(_btn_style_accent)
        self.btn_save_img.clicked.connect(self.save_analysis_image)

        toolbar.addWidget(_make_group([self.btn_export, self.btn_save_img]))

        # -- Spacer to push lang button to the right --
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        toolbar.addWidget(spacer)

        # -- Language button (standalone, right-aligned) --
        self.btn_lang = QPushButton("English")
        self.btn_lang.setStyleSheet(_btn_style_normal)
        self.btn_lang.clicked.connect(self._toggle_language)
        toolbar.addWidget(_make_group([self.btn_lang]))

        # ── Main layout ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left side: image display area ──
        left_widget = QWidget()
        left_widget.setStyleSheet("QWidget { background-color: #131620; }")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Image area + side navigation arrows
        image_area = QWidget()
        image_area.setStyleSheet("QWidget { background-color: #131620; }")
        image_h_layout = QHBoxLayout(image_area)
        image_h_layout.setContentsMargins(0, 0, 0, 0)
        image_h_layout.setSpacing(0)

        self.btn_prev_side = QPushButton("<")
        self.btn_prev_side.setFixedWidth(28)
        self.btn_prev_side.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.btn_prev_side.clicked.connect(self._prev_image)
        self.btn_prev_side.setStyleSheet("""
            QPushButton {
                background: transparent; color: #252a3c;
                border: none; font-size: 22px; font-weight: 300;
            }
            QPushButton:hover { color: #6eb4ff; }
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

        # ── Welcome page (shown when no image loaded) ──
        from PyQt5.QtWidgets import QStackedWidget, QListWidget, QListWidgetItem
        self._image_stack = QStackedWidget()
        self._welcome_page = QWidget()
        self._welcome_page.setStyleSheet("QWidget { background-color: #10131c; }")
        welcome_layout = QVBoxLayout(self._welcome_page)
        welcome_layout.setAlignment(Qt.AlignCenter)
        welcome_layout.setSpacing(0)

        # Title
        self._welcome_title = QLabel("FAST IHC Analyzer")
        self._welcome_title.setAlignment(Qt.AlignCenter)
        self._welcome_title.setStyleSheet(
            "font-size: 32px; font-weight: 300; color: #4a6a9a; "
            "border: none; background: transparent; padding: 0; "
            "letter-spacing: 4px;")
        welcome_layout.addWidget(self._welcome_title)

        # Subtitle
        self._welcome_subtitle = QLabel("请打开图像或文件夹开始分析")
        self._welcome_subtitle.setAlignment(Qt.AlignCenter)
        self._welcome_subtitle.setStyleSheet(
            "font-size: 14px; color: #3a4a68; border: none; "
            "background: transparent; padding: 6px 0 36px 0;")
        welcome_layout.addWidget(self._welcome_subtitle)

        # ── Action cards (vertical, centered) ──
        cards_container = QWidget()
        cards_container.setStyleSheet("background: transparent;")
        cards_container.setMaximumWidth(400)
        cards_layout = QVBoxLayout(cards_container)
        cards_layout.setSpacing(12)
        cards_layout.setContentsMargins(0, 0, 0, 0)

        # Card: Open Folder (primary)
        self._welcome_btn_folder = QPushButton("打开文件夹")
        self._welcome_btn_folder.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e55cc, stop:0.5 #2868e0, stop:1 #3580f8);
                border: none;
                border-radius: 12px;
                padding: 16px 40px;
                color: #ffffff;
                font-size: 17px;
                font-weight: 600;
                min-width: 280px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2868e0, stop:0.5 #3580f8, stop:1 #4a98ff);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1848b0, stop:0.5 #2058cc, stop:1 #2868e0);
            }
        """)
        self._welcome_btn_folder.clicked.connect(self.open_folder)
        self._welcome_btn_folder.setCursor(QCursor(Qt.PointingHandCursor))
        cards_layout.addWidget(self._welcome_btn_folder, 0, Qt.AlignCenter)

        # Card: Open Image (secondary / outline)
        self._welcome_btn_image = QPushButton("打开图像")
        self._welcome_btn_image.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #252e44;
                border-radius: 12px;
                padding: 14px 40px;
                color: #6080b0;
                font-size: 15px;
                font-weight: 500;
                min-width: 280px;
            }
            QPushButton:hover {
                border-color: #3a5080;
                color: #90b0e0;
                background-color: rgba(60, 100, 180, 0.06);
            }
            QPushButton:pressed {
                background-color: rgba(60, 100, 180, 0.12);
            }
        """)
        self._welcome_btn_image.clicked.connect(self.open_image)
        self._welcome_btn_image.setCursor(QCursor(Qt.PointingHandCursor))
        cards_layout.addWidget(self._welcome_btn_image, 0, Qt.AlignCenter)

        welcome_layout.addWidget(cards_container, 0, Qt.AlignCenter)

        # Recent folders section
        self._recent_label = QLabel("最近打开的文件夹")
        self._recent_label.setAlignment(Qt.AlignCenter)
        self._recent_label.setStyleSheet(
            "font-size: 13px; color: #3a4a68; border: none; "
            "background: transparent; padding-top: 32px; padding-bottom: 8px;")
        welcome_layout.addWidget(self._recent_label)

        self._recent_list = QListWidget()
        self._recent_list.setMaximumWidth(500)
        self._recent_list.setMaximumHeight(200)
        self._recent_list.setStyleSheet("""
            QListWidget {
                background-color: #141824;
                border: 1px solid #1a2035;
                border-radius: 10px;
                padding: 6px;
                font-size: 14px;
                color: #8090b0;
            }
            QListWidget::item {
                padding: 8px 14px;
                border-radius: 6px;
                border: none;
            }
            QListWidget::item:hover {
                background-color: rgba(80, 140, 255, 0.1);
                color: #b0c8f0;
            }
            QListWidget::item:selected {
                background-color: rgba(80, 140, 255, 0.15);
                color: #6eb4ff;
            }
        """)
        self._recent_list.itemDoubleClicked.connect(self._open_recent_folder)
        self._recent_list.setCursor(QCursor(Qt.PointingHandCursor))
        welcome_layout.addWidget(self._recent_list, 0, Qt.AlignCenter)

        self._load_recent_folders()

        self._image_stack.addWidget(self._welcome_page)  # index 0
        self._image_stack.addWidget(self.image_tabs)       # index 1
        self._image_stack.setCurrentIndex(0)

        image_h_layout.addWidget(self._image_stack)

        self.btn_next_side = QPushButton(">")
        self.btn_next_side.setFixedWidth(28)
        self.btn_next_side.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.btn_next_side.clicked.connect(self._next_image)
        self.btn_next_side.setStyleSheet("""
            QPushButton {
                background: transparent; color: #252a3c;
                border: none; font-size: 22px; font-weight: 300;
            }
            QPushButton:hover { color: #6eb4ff; }
        """)
        image_h_layout.addWidget(self.btn_next_side)

        left_layout.addWidget(image_area)
        splitter.addWidget(left_widget)

        # ── Right side: control panel ──
        right_widget = QWidget()
        right_widget.setMinimumWidth(360)
        right_widget.setStyleSheet("QWidget { background-color: #131620; }")
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("QScrollArea { background-color: #131620; border: none; }")
        right_inner = QWidget()
        right_inner.setStyleSheet("QWidget { background-color: #131620; }")
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
        self.result_text.setFont(QFont("SF Mono, Menlo, Consolas, monospace", 14))
        result_layout.addWidget(self.result_text, 1)  # stretch=1 auto-fill

        self.grp_result.setLayout(result_layout)
        right_inner_layout.addWidget(self.grp_result, 1)  # stretch=1 fill

        right_scroll.setWidget(right_inner)
        right_layout.addWidget(right_scroll)
        splitter.addWidget(right_widget)

        splitter.setSizes([850, 450])

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
        upper_widget.setStyleSheet("QWidget { background-color: #131620; }")
        upper_widget.setLayout(main_layout)
        vsplitter.addWidget(upper_widget)
        vsplitter.addWidget(self.batch_tab)
        vsplitter.setSizes([600, 250])

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(vsplitter)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        outer_widget = QWidget()
        outer_widget.setStyleSheet("QWidget { background-color: #131620; }")
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
        if folder:
            self._do_open_folder(folder)

    def _do_open_folder(self, folder):
        """Open a folder by path (used by both dialog and recent list)."""
        IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.svs'}
        paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        if not paths:
            QMessageBox.information(self, "提示", "所选文件夹中未找到图像文件")
            return
        self._save_recent_folder(folder)
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

        # Switch from welcome page to image tabs
        self._show_image_tabs()

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

    def _calculate_scores(self, dab=None, roi=None, positive_ratio=None, thresholds=None, roi_mask=None):
        """Calculate IHC scores (mirrors tiff/ihc_gui.py IHCAnalyzer logic).
        - Grayscale grading on masked image, thresholds controlled by sliders
        - Higher grayscale = stronger staining within positive regions
        - thresholds: (t_high, t_pos, t_low) three grayscale thresholds
        - roi_mask: optional numpy binary mask for freehand ROI
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

        # Get analysis region — freehand polygon mask or bounding rect
        if roi_mask is None and roi is None:
            roi_mask = self.canvas_original.get_roi_mask(dab.shape)

        if roi_mask is not None:
            # Freehand ROI: extract pixels within the polygon
            gray = dab[roi_mask > 0]
            is_en = hasattr(self, 'lang') and self.lang is self.LANG_EN
            n_roi = int(np.sum(roi_mask > 0))
            area_info = f"ROI (freehand, {n_roi:,} px)" if is_en else f"ROI (自由圈选, {n_roi:,} px)"
        elif roi is not None and not roi.isNull():
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

        # When very few positive pixels (<5%), cap intensity to avoid noise-driven inflation
        if positive_ratio < 0.05 and intensity_score > 1:
            intensity_score = 1
            intensity_label = 'Low Positive' if is_en else '弱阳性'

        intensity_basis = (f"Mean positive gray: {mean_intensity:.1f}"
                           if is_en else f"阳性区域平均灰度: {mean_intensity:.1f}")

        # Positive proportion score (1-4), based on HSV-detected positive_ratio
        # Adjusted thresholds: 10/25/50 for better sensitivity on low-expression markers
        pos_pct = positive_ratio * 100
        if pos_pct <= 10:
            proportion_score = 1
        elif pos_pct <= 25:
            proportion_score = 2
        elif pos_pct <= 50:
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
        Supports both freehand polygon ROI and rectangular ROI.
        """
        if self.rgb_image is None or self.dab_channel is None:
            return

        overlay = self.rgb_image.copy()
        dab = self.dab_channel
        hsv_mask_full = self.hsv_mask

        t_high = self.slider_strong.value()
        t_pos = self.slider_moderate.value()
        t_low = self.slider_weak.value()

        # Determine analysis region mask (full image, freehand, or rect)
        roi_mask = self.canvas_original.get_roi_mask(dab.shape)
        roi_rect = self.canvas_original.get_roi()

        if roi_mask is not None:
            # Freehand ROI: build 2D grading masks within the polygon
            in_roi = (roi_mask > 0)
            high_mask = in_roi & (dab >= t_high)
            pos_mask = in_roi & (dab >= t_pos) & (dab < t_high)
            low_mask = in_roi & (dab >= t_low) & (dab < t_pos)
            # Dim everything outside the ROI
            outside = ~in_roi
            overlay[outside] = (overlay[outside] * 0.4).astype(np.uint8)
            # Dim non-positive inside ROI
            if hsv_mask_full is not None:
                non_pos_in_roi = in_roi & (hsv_mask_full == 0)
                overlay[non_pos_in_roi] = (overlay[non_pos_in_roi] * 0.7).astype(np.uint8)
        elif roi_rect and not roi_rect.isNull():
            x, y, w, h = roi_rect.x(), roi_rect.y(), roi_rect.width(), roi_rect.height()
            x, y = max(0, x), max(0, y)
            w = min(w, dab.shape[1] - x)
            h = min(h, dab.shape[0] - y)
            roi_dab = dab[y:y+h, x:x+w]
            high_mask_r = (roi_dab >= t_high)
            pos_mask_r = (roi_dab >= t_pos) & (roi_dab < t_high)
            low_mask_r = (roi_dab >= t_low) & (roi_dab < t_pos)
            # Create full-size masks
            high_mask = np.zeros(dab.shape, dtype=bool)
            pos_mask = np.zeros(dab.shape, dtype=bool)
            low_mask = np.zeros(dab.shape, dtype=bool)
            high_mask[y:y+h, x:x+w] = high_mask_r
            pos_mask[y:y+h, x:x+w] = pos_mask_r
            low_mask[y:y+h, x:x+w] = low_mask_r
            if hsv_mask_full is not None:
                non_positive = (hsv_mask_full == 0)
                overlay[non_positive] = (overlay[non_positive] * 0.7).astype(np.uint8)
        else:
            # Full image
            high_mask = (dab >= t_high)
            pos_mask = (dab >= t_pos) & (dab < t_high)
            low_mask = (dab >= t_low) & (dab < t_pos)
            if hsv_mask_full is not None:
                non_positive = (hsv_mask_full == 0)
                overlay[non_positive] = (overlay[non_positive] * 0.7).astype(np.uint8)

        # Colorize by intensity grade
        alpha = 0.45
        if np.any(low_mask):
            overlay[low_mask] = (overlay[low_mask] * (1 - alpha) + np.array([102, 187, 106]) * alpha).astype(np.uint8)
        if np.any(pos_mask):
            overlay[pos_mask] = (overlay[pos_mask] * (1 - alpha) + np.array([255, 167, 38]) * alpha).astype(np.uint8)
        if np.any(high_mask):
            overlay[high_mask] = (overlay[high_mask] * (1 - alpha) + np.array([239, 83, 80]) * alpha).astype(np.uint8)

        self.score_mask = overlay
        self.canvas_score.set_image(overlay, is_rgb=True)
        self.image_tabs.setCurrentIndex(3)

    def _update_histogram(self):
        """Update grayscale histogram."""
        if self.dab_channel is None:
            return
        roi_mask = self.canvas_original.get_roi_mask(self.dab_channel.shape)
        if roi_mask is not None:
            data = self.dab_channel[roi_mask > 0]
        else:
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

    def _update_canvas_hints(self):
        """Update placeholder hints on welcome page."""
        L = self.lang
        self._welcome_subtitle.setText(L['empty_hint'])
        self._welcome_btn_image.setText(L['toolbar_open'])
        self._welcome_btn_folder.setText(L['toolbar_folder'])
        if L is self.LANG_ZH:
            self._recent_label.setText("最近打开的文件夹")
        else:
            self._recent_label.setText("Recent Folders")

    def _show_welcome(self):
        """Show the welcome page."""
        self._image_stack.setCurrentIndex(0)

    def _show_image_tabs(self):
        """Show the image tabs."""
        self._image_stack.setCurrentIndex(1)

    @staticmethod
    def _recent_folders_path():
        """Path to recent folders file."""
        config_dir = os.path.join(os.path.expanduser("~"), ".ihc_analyzer")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "recent_folders.txt")

    def _load_recent_folders(self):
        """Load recent folders from disk."""
        self._recent_list.clear()
        path = self._recent_folders_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                folders = [line.strip() for line in f if line.strip() and os.path.isdir(line.strip())]
            for folder in folders[:8]:
                # Show folder name + parent path
                name = os.path.basename(folder)
                parent = os.path.dirname(folder)
                from PyQt5.QtWidgets import QListWidgetItem
                item = QListWidgetItem(f"{name}    {parent}")
                item.setData(Qt.UserRole, folder)
                self._recent_list.addItem(item)
        if self._recent_list.count() == 0:
            self._recent_label.hide()
            self._recent_list.hide()
        else:
            self._recent_label.show()
            self._recent_list.show()

    def _save_recent_folder(self, folder):
        """Save a folder to the recent list (most recent first, max 8)."""
        path = self._recent_folders_path()
        folders = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                folders = [line.strip() for line in f if line.strip()]
        # Remove duplicates, add to front
        folder = os.path.abspath(folder)
        folders = [f for f in folders if f != folder]
        folders.insert(0, folder)
        folders = folders[:8]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(folders) + '\n')
        self._load_recent_folders()

    def _open_recent_folder(self, item):
        """Open a folder from the recent list."""
        folder = item.data(Qt.UserRole)
        if folder and os.path.isdir(folder):
            self._do_open_folder(folder)

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
        self.btn_lang.setText(L['lang_switch'])

        # Toolbar buttons
        self.btn_open.setText(L['toolbar_open'])
        self.btn_folder.setText(L['toolbar_folder'])
        # Navigation buttons keep < > symbols
        self.btn_roi.setText(L['toolbar_roi'])
        self.btn_clear_roi.setText(L['toolbar_clear_roi'])
        self.btn_analyze.setText(L['toolbar_analyze'])
        self.btn_batch_analyze.setText(L['toolbar_batch_analyze'])
        self.btn_export.setText(L['toolbar_export'])
        self.btn_save_img.setText(L['toolbar_save'])

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

        # Update canvas empty hints
        self._update_canvas_hints()

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

    def _on_roi_selected(self, roi_data):
        self.btn_roi.setChecked(False)
        # Sync freehand ROI to DAB canvas
        if isinstance(roi_data, list):
            self.canvas_dab._roi_points = list(roi_data)
            self.canvas_dab._update_display()
        self._update_histogram()
        if isinstance(roi_data, list):
            n_pts = len(roi_data)
            self.statusBar().showMessage(f"已选择ROI: 自由圈选 ({n_pts} 个点)")
        else:
            self.statusBar().showMessage(
                f"已选择ROI: ({roi_data.x()}, {roi_data.y()}) - "
                f"{roi_data.width()}x{roi_data.height()}"
            )

    def _clear_roi(self):
        self.canvas_original.clear_roi()
        self.canvas_dab.clear_roi()
        self.canvas_score.clear_roi()
        self._update_histogram()
        # Re-analyze without ROI to refresh score overlay
        if self.dab_channel is not None:
            results = self._calculate_scores()
            self._display_results(results)
            self._create_score_overlay(results)
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
    # Windows: set AppUserModelID so taskbar shows custom icon instead of Python's
    if platform.system() == 'Windows':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                'FastIHCAnalyzer.2.0')
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("FAST IHC Analyzer")

    # High DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Set app-level icon (taskbar / dock / window title bar)
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
