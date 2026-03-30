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
import imageio
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
    """批量分析结果表格"""
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
            # 临床判定列用颜色标注
            if col == 7:
                if 'Positive' in text or '阳性' in text:
                    item.setForeground(QColor('#ffa726'))
                else:
                    item.setForeground(QColor('#42a5f5'))
            self.setItem(row, col, item)


class IHCScorer(QMainWindow):
    """IHC评分分析主窗口"""
    CLINICAL_POSITIVE_THRESHOLD = 5.0

    # ── 中英文文本 ──
    LANG_ZH = {
        'title': 'IHC Score Analyzer - 免疫组化评分分析',
        'open': '打开图像(&O)', 'open_folder': '打开文件夹(&D)',
        'export': '导出结果(&E)', 'save_img': '保存分析图像(&S)',
        'toolbar_open': '打开图像', 'toolbar_folder': '打开文件夹',
        'toolbar_roi': '选择ROI', 'toolbar_clear_roi': '清除ROI',
        'toolbar_analyze': '分析', 'toolbar_batch_analyze': '批量分析',
        'toolbar_export': '导出CSV', 'toolbar_save': '保存图像',
        'grp_deconv': '颜色反卷积设置', 'stain_label': '染色方案:',
        'auto_wb': '自动白平衡',
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
        'title': 'IHC Score Analyzer',
        'open': 'Open Image(&O)', 'open_folder': 'Open Folder(&D)',
        'export': 'Export Results(&E)', 'save_img': 'Save Analysis Image(&S)',
        'toolbar_open': 'Open', 'toolbar_folder': 'Open Folder',
        'toolbar_roi': 'Select ROI', 'toolbar_clear_roi': 'Clear ROI',
        'toolbar_analyze': 'Analyze', 'toolbar_batch_analyze': 'Batch Analyze',
        'toolbar_export': 'Export CSV', 'toolbar_save': 'Save Image',
        'grp_deconv': 'Color Deconvolution', 'stain_label': 'Stain:',
        'auto_wb': 'Auto White Balance',
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

        # 数据
        self.original_image = None   # BGR
        self.rgb_image = None        # RGB
        self.dab_channel = None      # DAB通道 (masked image grayscale)
        self.hem_channel = None      # Hematoxylin通道 (preprocessed grayscale)
        self.score_mask = None       # 评分掩膜
        self.current_file = ""
        self.batch_files = []
        # tiff 逻辑新增
        self.preprocessed_image = None  # CLAHE预处理后的RGB图像
        self.masked_image = None        # HSV掩膜后的图像
        self.hsv_mask = None            # HSV二值掩膜
        self.positive_ratio = 0.0       # 阳性像素比例
        self.hsv_params = {
            'hue_low': 0, 'hue_high': 20,
            'saturation_low': 50, 'value_low': 50
        }

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

        # ── 工具栏 ──
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

        self.btn_roi = QPushButton("✂️ 选择ROI")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self._toggle_roi_mode)
        toolbar.addWidget(self.btn_roi)

        self.btn_clear_roi = QPushButton("↩️ 清除ROI")
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
        right_widget.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_inner = QWidget()
        right_inner_layout = QVBoxLayout(right_inner)

        # ── 反卷积设置 ──
        self.grp_deconv = QGroupBox("颜色反卷积设置")
        deconv_layout = QGridLayout()

        self.lbl_stain = QLabel("染色方案:")
        deconv_layout.addWidget(self.lbl_stain, 0, 0)
        self.stain_combo = QComboBox()
        stain_names = list(STAIN_VECTORS.keys())
        self.stain_combo.addItems(stain_names)
        he_idx = stain_names.index("H-E") if "H-E" in stain_names else 0
        self.stain_combo.setCurrentIndex(he_idx)
        deconv_layout.addWidget(self.stain_combo, 0, 1)

        self.chk_auto_balance = QCheckBox("自动白平衡")
        self.chk_auto_balance.setChecked(True)
        deconv_layout.addWidget(self.chk_auto_balance, 1, 0, 1, 2)

        self.grp_deconv.setLayout(deconv_layout)
        right_inner_layout.addWidget(self.grp_deconv)

        # ── 阈值设置 ──
        self.grp_thresh = QGroupBox("阈值设置 (灰度值)")
        thresh_layout = QGridLayout()

        self.threshold_info_label = QLabel()
        self.threshold_info_label.setStyleSheet("color: #888; font-size: 11px;")
        thresh_layout.addWidget(self.threshold_info_label, 0, 0, 1, 3)

        self.lbl_high_tag = QLabel("High+ <=")
        thresh_layout.addWidget(self.lbl_high_tag, 1, 0)
        self.slider_strong = QSlider(Qt.Horizontal)
        self.slider_strong.setRange(0, 255)
        self.slider_strong.setValue(160)
        self.slider_strong.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_strong, 1, 1)
        self.lbl_strong = QLabel("160")
        self.lbl_strong.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_strong, 1, 2)

        self.lbl_pos_tag = QLabel("Positive <=")
        thresh_layout.addWidget(self.lbl_pos_tag, 2, 0)
        self.slider_moderate = QSlider(Qt.Horizontal)
        self.slider_moderate.setRange(0, 255)
        self.slider_moderate.setValue(100)
        self.slider_moderate.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_moderate, 2, 1)
        self.lbl_moderate = QLabel("100")
        self.lbl_moderate.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_moderate, 2, 2)

        self.lbl_low_tag = QLabel("Low+ <=")
        thresh_layout.addWidget(self.lbl_low_tag, 3, 0)
        self.slider_weak = QSlider(Qt.Horizontal)
        self.slider_weak.setRange(0, 255)
        self.slider_weak.setValue(40)
        self.slider_weak.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_weak, 3, 1)
        self.lbl_weak = QLabel("40")
        self.lbl_weak.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_weak, 3, 2)

        # 预设
        preset_layout = QHBoxLayout()
        self.lbl_preset = QLabel("预设:")
        preset_layout.addWidget(self.lbl_preset)
        self.btn_preset_std = QPushButton("标准")
        self.btn_preset_std.setFixedHeight(28)
        self.btn_preset_std.clicked.connect(lambda: self._set_thresholds(160, 100, 40))
        preset_layout.addWidget(self.btn_preset_std)
        self.btn_preset_strict = QPushButton("严格")
        self.btn_preset_strict.setFixedHeight(28)
        self.btn_preset_strict.clicked.connect(lambda: self._set_thresholds(180, 120, 60))
        preset_layout.addWidget(self.btn_preset_strict)
        self.btn_preset_loose = QPushButton("宽松")
        self.btn_preset_loose.setFixedHeight(28)
        self.btn_preset_loose.clicked.connect(lambda: self._set_thresholds(140, 80, 20))
        preset_layout.addWidget(self.btn_preset_loose)
        thresh_layout.addLayout(preset_layout, 4, 0, 1, 3)

        self.lbl_bg_tag = QLabel("背景排除 >=")
        thresh_layout.addWidget(self.lbl_bg_tag, 5, 0)
        self.slider_tissue = QSlider(Qt.Horizontal)
        self.slider_tissue.setRange(0, 255)
        self.slider_tissue.setValue(236)
        self.slider_tissue.valueChanged.connect(self._on_threshold_changed)
        thresh_layout.addWidget(self.slider_tissue, 5, 1)
        self.lbl_tissue = QLabel("236")
        self.lbl_tissue.setMinimumWidth(30)
        thresh_layout.addWidget(self.lbl_tissue, 5, 2)

        self.grp_thresh.setLayout(thresh_layout)
        right_inner_layout.addWidget(self.grp_thresh)

        # ── DAB直方图 ──
        self.grp_hist = QGroupBox("DAB通道直方图")
        hist_layout = QVBoxLayout()
        self.histogram = HistogramWidget()
        hist_layout.addWidget(self.histogram)
        self.grp_hist.setLayout(hist_layout)
        right_inner_layout.addWidget(self.grp_hist)

        # ── 评分结果 ──
        self.grp_result = QGroupBox("评分结果")
        result_layout = QVBoxLayout()

        self.pie_chart = ScorePieChart()
        result_layout.addWidget(self.pie_chart)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(260)
        self.result_text.setFont(QFont("Courier", 12))
        result_layout.addWidget(self.result_text)

        self.grp_result.setLayout(result_layout)
        right_inner_layout.addWidget(self.grp_result)

        right_inner_layout.addStretch()
        right_scroll.setWidget(right_inner)
        right_layout.addWidget(right_scroll)
        splitter.addWidget(right_widget)

        splitter.setSizes([900, 400])

        # ── 底部: 批量结果标签页（可拖拽缩放） ──
        self.batch_tab = QTabWidget()
        self.batch_table = BatchResultTable()
        self.batch_tab.addTab(self.batch_table, "批量分析结果")
        self.batch_tab.setMinimumHeight(80)
        self.batch_tab.hide()

        # 垂直分割器：上方图像+控制面板，下方批量表格
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

        # ── 状态栏 ──
        self.statusBar().showMessage("就绪 - 请打开一张IHC染色图像开始分析")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self._on_threshold_changed()

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
        """防御式加载：优先 imageio（更好的 TIFF 支持），回退 cv2"""
        img = None
        try:
            img = imageio.imread(path)
            if img is None:
                raise ValueError("imageio.imread returned None")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # imageio 返回 RGB，转为 BGR 以兼容后续流程
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

        self.canvas_original.set_image(self.rgb_image)
        self.canvas_original.clear_roi()

        # 自动进行预处理 + HSV检测
        self._perform_deconvolution()

        self.setWindowTitle(f"IHC Score Analyzer - {os.path.basename(path)}")
        self.statusBar().showMessage(f"已加载: {path}  |  尺寸: {img.shape[1]}×{img.shape[0]}")

    # ─── 预处理 + HSV 阳性检测（复刻 tiff 逻辑） ──────────────────
    def _perform_deconvolution(self):
        """预处理 + HSV 阳性区域检测（遵循 tiff/ihc_gui.py 的 IHCAnalyzer 逻辑）"""
        if self.rgb_image is None:
            return

        # ── Step 1: 预处理 (GaussianBlur + CLAHE on LAB L-channel) ──
        blurred = cv2.GaussianBlur(self.rgb_image, (3, 3), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_ch)
        self.preprocessed_image = cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2RGB)

        # ── Step 2: HSV 阳性区域检测 ──
        hsv = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_RGB2HSV)
        lower = np.array([self.hsv_params['hue_low'],
                          self.hsv_params['saturation_low'],
                          self.hsv_params['value_low']])
        upper = np.array([self.hsv_params['hue_high'], 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        self.hsv_mask = mask
        self.masked_image = cv2.bitwise_and(
            self.preprocessed_image, self.preprocessed_image, mask=mask)
        total_pixels = mask.size
        positive_pixels = cv2.countNonZero(mask)
        self.positive_ratio = positive_pixels / total_pixels if total_pixels else 0

        # ── 为 UI 兼容生成灰度通道 ──
        # dab_channel: masked image 灰度（阳性区域有值，其余为0）
        self.dab_channel = cv2.cvtColor(self.masked_image, cv2.COLOR_RGB2GRAY)
        # hem_channel: 预处理后图像灰度
        self.hem_channel = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_RGB2GRAY)

        # ── 显示通道图像 (使用伪彩色) ──
        dab_color = cv2.applyColorMap(255 - self.dab_channel, cv2.COLORMAP_HOT)
        hem_color = cv2.applyColorMap(255 - self.hem_channel, cv2.COLORMAP_BONE)

        self.canvas_dab.set_image(cv2.cvtColor(dab_color, cv2.COLOR_BGR2RGB))
        self.canvas_hem.set_image(cv2.cvtColor(hem_color, cv2.COLOR_BGR2RGB))

        # 更新直方图
        self._update_histogram()

    @staticmethod
    def _preprocess_rgb(rgb):
        """对 RGB 图像执行 GaussianBlur + CLAHE 预处理（批量分析用）"""
        blurred = cv2.GaussianBlur(rgb, (3, 3), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2RGB)

    @staticmethod
    def _detect_positive_hsv(preprocessed_rgb, params):
        """HSV 阳性区域检测，返回 (mask, masked_image, positive_ratio)"""
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
        """获取并规范化当前灰度阈值"""
        t_high = self.slider_strong.value()
        t_pos = self.slider_moderate.value()
        t_low = self.slider_weak.value()
        t_tissue = self.slider_tissue.value()
        return t_high, t_pos, t_low, t_tissue

    def _update_threshold_info(self):
        """更新阈值说明文本（HSV检测 + 灰度强度分级）"""
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

    # ─── 分析 ─────────────────────────────────────────────────────
    def analyze_current(self):
        if self.dab_channel is None:
            QMessageBox.warning(self, "提示", "请先打开一张IHC染色图像")
            return
        results = self._calculate_scores()
        self._display_results(results)
        self._create_score_overlay(results)

    def _calculate_scores(self, dab=None, roi=None, positive_ratio=None):
        """计算IHC评分（复刻 tiff/ihc_gui.py 的 IHCAnalyzer 逻辑）
        - 基于 masked image 灰度分级
        - 灰度值越高 = 在阳性区域内染色越强
        - gray < 40: 阴性 (包括被HSV掩膜排除的黑色像素)
        - 40 <= gray < 100: 低阳性
        - 100 <= gray < 160: 阳性
        - gray >= 160: 强阳性
        """
        if dab is None:
            dab = self.dab_channel
        if positive_ratio is None:
            positive_ratio = self.positive_ratio

        # 获取分析区域
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
            area_info = "全图"

        total_pixels = int(gray.size)

        if total_pixels == 0:
            return {
                'negative': 100.0, 'low_pos': 0.0, 'positive': 0.0, 'high_pos': 0.0,
                'h_score': 0.0, 'positive_rate': 0.0,
                'intensity_score': 0, 'intensity_label': '阴性',
                'intensity_basis': '无像素',
                'mean_positive_gray': None,
                'proportion_score': 1, 'ihc_score': 0,
                'clinical': 'Negative',
                'clinical_detail': '阴性 [-]',
                'total_pixels': 0, 'tissue_pixels': 0,
                'background_pixels': 0,
                'area_info': area_info
            }

        # ── 强度分级（与 tiff IHCAnalyzer.analyze_intensity_levels 一致）──
        # masked image 中: 被掩膜排除的像素 = 0(黑色), 阳性像素保留原灰度
        n_high = int(np.sum(gray >= 160))
        n_pos  = int(np.sum((gray >= 100) & (gray < 160)))
        n_low  = int(np.sum((gray >= 40) & (gray < 100)))
        n_neg  = int(np.sum(gray < 40))

        pct_high = n_high / total_pixels * 100
        pct_pos  = n_pos / total_pixels * 100
        pct_low  = n_low / total_pixels * 100
        pct_neg  = n_neg / total_pixels * 100

        total_pos = pct_high + pct_pos + pct_low
        score_label = 'Positive' if total_pos > 5 else 'Negative'

        # H-Score (保留兼容)
        h_score = 1 * pct_low + 2 * pct_pos + 3 * pct_high

        # ── 临床评分（与 tiff IHCAnalyzer.calculate_clinical_scores 一致）──
        # 阳性像素的平均灰度（gray > 0 排除被掩膜遮盖的黑色像素）
        positive_gray = gray[gray > 0]
        mean_intensity = float(np.mean(positive_gray)) if positive_gray.size else 0

        # 染色强度评分 (0-3)
        if mean_intensity < 40:
            intensity_score = 0
            intensity_label = '阴性'
        elif mean_intensity < 100:
            intensity_score = 1
            intensity_label = '弱阳性'
        elif mean_intensity < 160:
            intensity_score = 2
            intensity_label = '阳性'
        else:
            intensity_score = 3
            intensity_label = '强阳性'

        intensity_basis = f"阳性区域平均灰度: {mean_intensity:.1f}"

        # 阳性比例评分 (1-4)，基于 HSV 检测的 positive_ratio
        pos_pct = positive_ratio * 100
        if pos_pct <= 25:
            proportion_score = 1
        elif pos_pct <= 50:
            proportion_score = 2
        elif pos_pct <= 75:
            proportion_score = 3
        else:
            proportion_score = 4

        # IHC 评分 = 强度 x 比例 (0-12)
        ihc_score = intensity_score * proportion_score

        # 临床判定
        if score_label == 'Negative':
            clinical = 'Negative'
            clinical_detail = '阴性 [-]'
        elif ihc_score <= 3:
            clinical = 'Positive'
            clinical_detail = '弱阳性 [+]'
        elif ihc_score <= 6:
            clinical = 'Positive'
            clinical_detail = '阳性 [++]'
        else:
            clinical = 'Positive'
            clinical_detail = '强阳性 [+++]'

        # 组织/背景像素统计
        tissue_pixels = int(np.sum(gray > 0))
        background_pixels = int(np.sum(gray == 0))

        # 分级掩膜（用于叠加显示）
        high_mask = (gray >= 160)
        pos_mask  = (gray >= 100) & (gray < 160)
        low_mask  = (gray >= 40) & (gray < 100)
        neg_mask  = (gray < 40)

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
        """显示评分结果"""
        t_high, t_pos, t_low, t_tissue = self._get_threshold_values()
        negative_upper = max(t_low + 1, t_tissue - 1)
        text = f"""{'='*42}
  IHC 评分报告
{'='*42}
  文件: {os.path.basename(self.current_file)}
  区域: {results['area_info']}
  总像素: {results['total_pixels']:,} px
  组织像素: {results['tissue_pixels']:,} px
  背景像素: {results['background_pixels']:,} px
{'_'*42}
  阴性 ({t_low + 1}-{negative_upper}): {results['negative']:6.1f}%
  弱阳性 ({t_pos + 1}-{t_low}): {results['low_pos']:6.1f}%
  阳性 ({t_high + 1}-{t_pos}): {results['positive']:6.1f}%
  强阳性 (0-{t_high}): {results['high_pos']:6.1f}%
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

        # 更新饼图
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

        # Negative 中的纯背景区域(原图灰度极高)稍微压暗以区分
        if self.rgb_image is not None:
            if roi and not roi.isNull():
                img_g = cv2.cvtColor(self.rgb_image[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
            else:
                img_g = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2GRAY)
            bg = img_g >= 236
            overlay[bg] = (overlay[bg] * 0.5).astype(np.uint8)

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
            data = self.dab_channel[y:y+h, x:x+w]
        else:
            data = self.dab_channel

        # 灰度值阈值线: High+, Positive, Low+
        thresholds = [
            self.slider_strong.value(),   # High+ boundary
            self.slider_moderate.value(),  # Positive boundary
            self.slider_weak.value(),      # Low+ boundary
        ]
        self.histogram.plot_histogram(
            data, "灰度分布",
            thresholds, colors=['#ef5350', '#ffa726', '#66bb6a']
        )

    # ─── 阈值变化 ─────────────────────────────────────────────────
    def _on_threshold_changed(self):
        # 确保阈值逻辑: 强阳性 < 阳性 < 弱阳性 < 背景
        s = min(self.slider_strong.value(), 252)
        if s != self.slider_strong.value():
            self.slider_strong.blockSignals(True)
            self.slider_strong.setValue(s)
            self.slider_strong.blockSignals(False)

        m = min(self.slider_moderate.value(), 253)
        if m <= s:
            m = s + 1
        if m != self.slider_moderate.value():
            self.slider_moderate.blockSignals(True)
            self.slider_moderate.setValue(m)
            self.slider_moderate.blockSignals(False)

        w = min(self.slider_weak.value(), 254)
        if w <= m:
            w = m + 1
        if w != self.slider_weak.value():
            self.slider_weak.blockSignals(True)
            self.slider_weak.setValue(w)
            self.slider_weak.blockSignals(False)

        t = self.slider_tissue.value()
        if t <= w:
            t = w + 1
        t = min(t, 255)
        if t != self.slider_tissue.value():
            self.slider_tissue.blockSignals(True)
            self.slider_tissue.setValue(t)
            self.slider_tissue.blockSignals(False)

        self.lbl_strong.setText(str(self.slider_strong.value()))
        self.lbl_moderate.setText(str(self.slider_moderate.value()))
        self.lbl_weak.setText(str(self.slider_weak.value()))
        self.lbl_tissue.setText(str(self.slider_tissue.value()))

        self._update_threshold_info()
        self._update_histogram()

    def _set_thresholds(self, high, pos, low):
        self.slider_strong.setValue(high)
        self.slider_moderate.setValue(pos)
        self.slider_weak.setValue(low)

    # ─── 语言切换 ──────────────────────────────────────────────────
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

        # 工具栏按钮
        self.btn_open.setText("📂 " + L['toolbar_open'])
        self.btn_folder.setText("📂 " + L['toolbar_folder'])
        self.btn_roi.setText("✂️ " + L['toolbar_roi'])
        self.btn_clear_roi.setText("↩️ " + L['toolbar_clear_roi'])
        self.btn_analyze.setText("▶ " + L['toolbar_analyze'])
        self.btn_batch_analyze.setText("▶▶ " + L['toolbar_batch_analyze'])
        self.btn_export.setText("💾 " + L['toolbar_export'])
        self.btn_save_img.setText("🖼 " + L['toolbar_save'])

        # 菜单项
        self.act_open.setText(L['open'])
        self.act_folder.setText(L['open_folder'])
        self.act_export.setText(L['export'])
        self.act_save_img.setText(L['save_img'])

        # 右侧面板 - GroupBox 标题
        self.grp_deconv.setTitle(L['grp_deconv'])
        self.grp_thresh.setTitle(L['grp_thresh'])
        self.grp_hist.setTitle(L['grp_hist'])
        self.grp_result.setTitle(L['grp_result'])

        # 右侧面板 - 标签
        self.lbl_stain.setText(L['stain_label'])
        self.chk_auto_balance.setText(L['auto_wb'])
        self.lbl_high_tag.setText(L['lbl_high'])
        self.lbl_pos_tag.setText(L['lbl_pos'])
        self.lbl_low_tag.setText(L['lbl_low'])
        self.lbl_preset.setText(L['preset_label'])
        self.btn_preset_std.setText(L['preset_std'])
        self.btn_preset_strict.setText(L['preset_strict'])
        self.btn_preset_loose.setText(L['preset_loose'])
        self.lbl_bg_tag.setText(L['bg_label'])

        # 图像标签页
        self.image_tabs.setTabText(0, L['tab_original'])
        self.image_tabs.setTabText(1, L['tab_dab'])
        self.image_tabs.setTabText(2, L['tab_hem'])
        self.image_tabs.setTabText(3, L['tab_score'])

        # 批量表格
        self.batch_tab.setTabText(0, L['batch_tab'])
        for col, header in enumerate(L['table_headers']):
            self.batch_table.setHorizontalHeaderItem(col, QTableWidgetItem(header))

        # 刷新阈值说明文字
        self._update_threshold_info()

        self.statusBar().showMessage(L['status_ready'])

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

            # 复刻 tiff 逻辑: 预处理 + HSV 检测
            preprocessed = self._preprocess_rgb(rgb)
            _mask, masked_img, pos_ratio = self._detect_positive_hsv(
                preprocessed, self.hsv_params)
            dab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

            results = self._calculate_scores(
                dab=dab, roi=None, positive_ratio=pos_ratio)

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
                        '图片名称', '总像素', '高强阳(%)', '中阳(%)', '低阳(%)',
                        '阴性(%)', '临床判定', '强度评分', '比例评分', 'IHC评分'
                    ])
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
