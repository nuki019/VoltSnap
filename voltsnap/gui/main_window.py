"""VoltSnap 主窗口 — 三栏布局"""
from __future__ import annotations

import csv
import json
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
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from voltsnap.gui.image_panel import ImagePanel
from voltsnap.gui.circuit_panel import CircuitPanel
from voltsnap.gui.schematic_editor import SchematicEditor
from voltsnap.gui.simulation_panel import SimulationPanel
from voltsnap.gui.worker import InferenceWorker, SimulationWorker
from voltsnap.models import SimulationResult

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
        self._last_recognition_result = None
        self._last_simulation_result: SimulationResult | None = None
        self._syncing_component_change = False

        # 初始化 UI
        self._init_panels()
        self._init_toolbar()
        self._init_menubar()
        self._init_statusbar()
        self._init_log_dock()
        self._connect_selection()

    # ── UI 初始化 ─────────────────────────────────────────────────────

    def _init_menubar(self):
        """初始化菜单栏，提供导出等操作的快捷入口"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction(self.action_open)
        file_menu.addSeparator()

        # 导出子菜单 — 提升导出功能可发现性
        export_menu = file_menu.addMenu("导出(&E)")
        export_menu.addAction(self.action_export)
        export_menu.addAction(self.action_export_recognition)
        export_menu.addAction(self.action_export_simulation)

    def _init_panels(self):
        """初始化三栏面板，左栏使用 Tab 切换图像视图和原理图"""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左栏：Tab 切换图像面板 / 原理图编辑器
        self.image_panel = ImagePanel()
        self.schematic_editor = SchematicEditor()

        left_tabs = QTabWidget()
        left_tabs.addTab(self.image_panel, "原始图像")
        left_tabs.addTab(self.schematic_editor, "原理图")
        splitter.addWidget(left_tabs)

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

        toolbar.addSeparator()

        # 导出网表
        self.action_export = QAction("导出网表", self)
        self.action_export.setShortcut("Ctrl+E")
        self.action_export.triggered.connect(self._on_export_netlist)
        toolbar.addAction(self.action_export)

        # 导出识别结果
        self.action_export_recognition = QAction("导出识别", self)
        self.action_export_recognition.setShortcut("Ctrl+Shift+E")
        self.action_export_recognition.triggered.connect(self._on_export_recognition)
        toolbar.addAction(self.action_export_recognition)

        # 导出仿真结果
        self.action_export_simulation = QAction("导出仿真", self)
        self.action_export_simulation.triggered.connect(self._on_export_simulation)
        toolbar.addAction(self.action_export_simulation)

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

    def _connect_selection(self):
        """连接图像面板、原理图编辑器与电路面板的双向选择联动"""
        self.image_panel.component_selected.connect(self._on_image_component_selected)
        self.circuit_panel.component_selected.connect(self._on_table_component_selected)
        self.circuit_panel.component_changed.connect(self._on_table_component_changed)
        self.schematic_editor.component_selected.connect(self._on_schematic_component_selected)
        self.schematic_editor.component_changed.connect(self._on_schematic_component_changed)

    # ── 事件处理 ──────────────────────────────────────────────────────

    def _on_image_component_selected(self, ref):
        """图像框点击 -> 选中表格行 + 原理图 + 更新状态栏"""
        self.circuit_panel.select_component(ref)
        self.schematic_editor.select_component(ref)
        self._update_status_for_component(ref)

    def _on_table_component_selected(self, ref):
        """表格行选择 -> 高亮图像框 + 原理图 + 更新状态栏"""
        self.image_panel.select_component(ref)
        self.schematic_editor.select_component(ref)
        self._update_status_for_component(ref)

    def _on_schematic_component_selected(self, ref):
        """原理图点击 -> 选中表格行 + 图像框 + 更新状态栏"""
        self.circuit_panel.select_component(ref)
        self.image_panel.select_component(ref)
        self._update_status_for_component(ref)

    def _on_table_component_changed(self, old_ref: str, updates: dict):
        if self._syncing_component_change:
            return
        self._syncing_component_change = True
        try:
            self.schematic_editor.refresh_component(old_ref, updates)
            new_ref = updates.get("ref", old_ref)
            if self.schematic_editor._selected_ref == new_ref:
                self._update_status_for_component(new_ref)
        finally:
            self._syncing_component_change = False

    def _on_schematic_component_changed(self, old_ref: str, updates: dict):
        if self._syncing_component_change:
            return
        self._syncing_component_change = True
        try:
            self.circuit_panel.update_component(old_ref, updates)
            new_ref = updates.get("ref", old_ref)
            self.circuit_panel.select_component(new_ref)
            self.image_panel.select_component(new_ref)
            self._update_status_for_component(new_ref)
        finally:
            self._syncing_component_change = False

    def _update_status_for_component(self, ref: str | None):
        """状态栏显示所选元件信息"""
        if ref is None:
            self.status_bar.showMessage("就绪 — 导入电路图开始识别")
            return
        comps = self.circuit_panel.get_components()
        comp = next((c for c in comps if c.get("ref") == ref), None)
        if comp:
            t = comp.get("type", "")
            display = self.circuit_panel.table_model.TYPE_DISPLAY.get(t, t)
            value = comp.get("value", "")
            conf = comp.get("confidence", 0)
            self.status_bar.showMessage(
                f"选中: {ref} | 类型: {display} | 值: {value} | 置信度: {conf:.0%}"
            )
        else:
            self.status_bar.showMessage(f"选中: {ref}")

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

        self._last_recognition_result = result

        # 更新图像面板（叠加识别框）
        self.image_panel.overlay_detections(result.detections)

        # 更新电路面板
        self.circuit_panel.load_components(result.bound_components)

        # 更新原理图编辑器
        self.schematic_editor.load_components(result.bound_components)

        # 更新仿真面板（网表）
        self.simulation_panel.set_netlist(result.netlist)

        self.log(f"识别完成: {len(result.components)} 个元件")
        self.status_bar.showMessage(
            f"识别完成: {len(result.detections)} 个检测, "
            f"{len(result.components)} 个绑定元件  |  可通过 文件→导出 保存结果"
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

        self._last_simulation_result = result

        if result.success:
            self.simulation_panel.show_results(result)
            self.schematic_editor.show_simulation_result(result)
            self.log(f"仿真成功: {result.node_voltages}")
            self.status_bar.showMessage("仿真完成  |  可通过 文件→导出 保存结果")
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
        if not path:
            return

        try:
            Path(path).write_text(netlist, encoding="utf-8")
            self.log(f"网表已导出: {path}")
            self.status_bar.showMessage(f"网表已导出: {Path(path).name}")
        except OSError as e:
            self.log(f"导出失败: {e}")
            QMessageBox.critical(self, "导出错误", f"无法写入文件:\n{e}")

    def _on_export_recognition(self):
        """导出识别结果为 JSON"""
        if self._last_recognition_result is None:
            QMessageBox.warning(self, "提示", "没有可导出的识别结果，请先运行识别")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出识别结果",
            "recognition_result.json",
            "JSON 文件 (*.json);;所有文件 (*)",
        )
        if not path:
            return

        try:
            result = self._last_recognition_result
            export_data = {
                "image_path": result.image_path,
                "success": result.success,
                "error_message": result.error_message,
                "components": [
                    {
                        "ref": c.ref,
                        "type": c.type,
                        "value": c.value,
                        "pins": [
                            {"name": p.name, "position": p.position, "pixel_position": p.pixel_position}
                            for p in c.pins
                        ],
                    }
                    for c in result.components
                ],
                "detections": result.detections,
                "ocr_results": result.ocr_results,
                "pin_to_net": {k: v for k, v in result.pin_to_net.items()},
                "netlist": result.netlist,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            self.log(f"识别结果已导出: {path}")
            self.status_bar.showMessage(f"识别结果已导出: {Path(path).name}")
        except (OSError, TypeError) as e:
            self.log(f"导出失败: {e}")
            QMessageBox.critical(self, "导出错误", f"无法导出识别结果:\n{e}")

    def _on_export_simulation(self):
        """导出仿真结果为 JSON 或 CSV"""
        if self._last_simulation_result is None:
            QMessageBox.warning(self, "提示", "没有可导出的仿真结果，请先运行仿真")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出仿真结果",
            "simulation_result.json",
            "JSON 文件 (*.json);;CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return

        try:
            result = self._last_simulation_result
            if path.endswith(".csv"):
                self._export_simulation_csv(path, result)
            else:
                self._export_simulation_json(path, result)
            self.log(f"仿真结果已导出: {path}")
            self.status_bar.showMessage(f"仿真结果已导出: {Path(path).name}")
        except (OSError, TypeError) as e:
            self.log(f"导出失败: {e}")
            QMessageBox.critical(self, "导出错误", f"无法导出仿真结果:\n{e}")

    @staticmethod
    def _export_simulation_json(path: str, result: SimulationResult):
        """导出仿真结果为 JSON"""
        export_data = {
            "success": result.success,
            "error_message": result.error_message,
            "node_voltages": result.node_voltages,
            "branch_currents": result.branch_currents,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _export_simulation_csv(path: str, result: SimulationResult):
        """导出仿真结果为 CSV"""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["类型", "名称", "值"])
            for node, voltage in sorted(result.node_voltages.items()):
                writer.writerow(["电压", node, f"{voltage:.6f}"])
            for branch, current in sorted(result.branch_currents.items()):
                writer.writerow(["电流", branch, f"{current:.6e}"])

    # ── 工具方法 ──────────────────────────────────────────────────────

    def log(self, message: str):
        """向日志面板追加消息"""
        self.log_text.appendPlainText(message)
        logger.info(message)
