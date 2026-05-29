"""VoltSnap 主窗口 — 三栏布局"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from voltsnap.gui.image_panel import ImagePanel
from voltsnap.gui.circuit_panel import CircuitPanel
from voltsnap.gui.simulation_panel import SimulationPanel
from voltsnap.gui.worker import InferenceWorker, SimulationWorker

logger = logging.getLogger("voltsnap.gui")


class MainWindow(QMainWindow):
    """
    VoltSnap 主窗口。

    三栏布局：
    - 左栏：原始图片 + 识别框叠加
    - 中栏：电路结构 + 可编辑参数
    - 右栏：仿真设置 + 网表 + 波形
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoltSnap — 电路图识别与仿真")
        self.setMinimumSize(1200, 700)
        self.resize(1600, 900)

        # 状态
        self._current_image_path: str | None = None
        self._inference_worker: InferenceWorker | None = None
        self._sim_worker: SimulationWorker | None = None

        # 初始化 UI
        self._init_panels()
        self._init_toolbar()
        self._init_statusbar()
        self._init_log_dock()

    # ── UI 初始化 ─────────────────────────────────────────────────────

    def _init_panels(self):
        """初始化三栏面板"""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左栏：图像面板
        self.image_panel = ImagePanel()
        splitter.addWidget(self.image_panel)

        # 中栏：电路面板
        self.circuit_panel = CircuitPanel()
        splitter.addWidget(self.circuit_panel)

        # 右栏：仿真面板
        self.simulation_panel = SimulationPanel()
        splitter.addWidget(self.simulation_panel)

        # 比例 4:3:3
        splitter.setSizes([600, 450, 450])

        self.setCentralWidget(splitter)

    def _init_toolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # 导入图片
        self.action_open = QAction("导入图片", self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.triggered.connect(self._on_open_image)
        toolbar.addAction(self.action_open)

        toolbar.addSeparator()

        # 运行识别
        self.action_recognize = QAction("运行识别", self)
        self.action_recognize.setShortcut("Ctrl+R")
        self.action_recognize.triggered.connect(self._on_run_recognition)
        toolbar.addAction(self.action_recognize)

        # 运行仿真
        self.action_simulate = QAction("运行仿真", self)
        self.action_simulate.setShortcut("Ctrl+Shift+R")
        self.action_simulate.triggered.connect(self._on_run_simulation)
        toolbar.addAction(self.action_simulate)

        toolbar.addSeparator()

        # 导出网表
        self.action_export = QAction("导出网表", self)
        self.action_export.setShortcut("Ctrl+E")
        self.action_export.triggered.connect(self._on_export_netlist)
        toolbar.addAction(self.action_export)

    def _init_statusbar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("就绪 — 导入电路图开始识别")

    def _init_log_dock(self):
        """初始化日志面板"""
        dock = QDockWidget("日志", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)
        dock.setWidget(self.log_text)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    # ── 事件处理 ──────────────────────────────────────────────────────

    def _on_open_image(self):
        """导入图片"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择电路图",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*)",
        )
        if not path:
            return

        self._current_image_path = path
        self.image_panel.load_image(path)
        self.log(f"已导入: {path}")
        self.status_bar.showMessage(f"已加载: {Path(path).name}")

    def _on_run_recognition(self):
        """运行识别"""
        if not self._current_image_path:
            QMessageBox.warning(self, "提示", "请先导入图片")
            return

        self.log("开始识别...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.action_recognize.setEnabled(False)

        self._inference_worker = InferenceWorker(self._current_image_path)
        self._inference_worker.finished.connect(self._on_recognition_done)
        self._inference_worker.error.connect(self._on_recognition_error)
        self._inference_worker.start()

    def _on_recognition_done(self, result):
        """识别完成"""
        self.progress_bar.setVisible(False)
        self.action_recognize.setEnabled(True)

        # 更新图像面板（叠加识别框）
        self.image_panel.overlay_detections(result.detections)

        # 更新电路面板
        self.circuit_panel.load_components(result.bound_components)

        # 更新仿真面板（网表）
        self.simulation_panel.set_netlist(result.netlist)

        self.log(f"识别完成: {len(result.components)} 个元件")
        self.status_bar.showMessage(
            f"识别完成: {len(result.detections)} 个检测, "
            f"{len(result.components)} 个绑定元件"
        )

    def _on_recognition_error(self, error_msg):
        """识别失败"""
        self.progress_bar.setVisible(False)
        self.action_recognize.setEnabled(True)
        self.log(f"识别失败: {error_msg}")
        QMessageBox.critical(self, "识别错误", error_msg)

    def _on_run_simulation(self):
        """运行仿真"""
        netlist = self.simulation_panel.get_netlist()
        if not netlist.strip():
            QMessageBox.warning(self, "提示", "没有可仿真的网表")
            return

        self.log("开始仿真...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.action_simulate.setEnabled(False)

        self._sim_worker = SimulationWorker(netlist)
        self._sim_worker.finished.connect(self._on_simulation_done)
        self._sim_worker.error.connect(self._on_simulation_error)
        self._sim_worker.start()

    def _on_simulation_done(self, result):
        """仿真完成"""
        self.progress_bar.setVisible(False)
        self.action_simulate.setEnabled(True)

        if result.success:
            self.simulation_panel.show_results(result)
            self.log(f"仿真成功: {result.node_voltages}")
        else:
            self.log(f"仿真失败: {result.error_message}")
            QMessageBox.warning(self, "仿真错误", result.error_message or "未知错误")

    def _on_simulation_error(self, error_msg):
        """仿真异常"""
        self.progress_bar.setVisible(False)
        self.action_simulate.setEnabled(True)
        self.log(f"仿真异常: {error_msg}")

    def _on_export_netlist(self):
        """导出网表"""
        netlist = self.simulation_panel.get_netlist()
        if not netlist.strip():
            QMessageBox.warning(self, "提示", "没有可导出的网表")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出网表",
            "circuit.cir",
            "SPICE 网表 (*.cir *.sp *.net);;所有文件 (*)",
        )
        if path:
            Path(path).write_text(netlist, encoding="utf-8")
            self.log(f"网表已导出: {path}")

    # ── 工具方法 ──────────────────────────────────────────────────────

    def log(self, message: str):
        """向日志面板追加消息"""
        self.log_text.appendPlainText(message)
        logger.info(message)
