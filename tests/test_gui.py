"""GUI 模块测试（无需显示器）"""
import json
import sys
from pathlib import Path

import pytest

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """创建 QApplication 实例"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestMainWindow:
    def test_window_creation(self, qapp):
        from voltsnap.gui.main_window import MainWindow
        window = MainWindow()
        assert window.windowTitle()
        assert window.width() >= 1200
        assert window.height() >= 700

    def test_panels_exist(self, qapp):
        from voltsnap.gui.main_window import MainWindow
        window = MainWindow()
        assert window.image_panel is not None
        assert window.circuit_panel is not None
        assert window.simulation_panel is not None

    def test_toolbar_actions(self, qapp):
        from voltsnap.gui.main_window import MainWindow
        window = MainWindow()
        assert window.action_open is not None
        assert window.action_recognize is not None
        assert window.action_simulate is not None
        assert window.action_export is not None
        assert window.action_export_recognition is not None
        assert window.action_export_simulation is not None

    def test_menubar_has_file_menu(self, qapp):
        """菜单栏应包含文件菜单和导出子菜单"""
        from voltsnap.gui.main_window import MainWindow
        window = MainWindow()
        menubar = window.menuBar()
        actions = [a.text() for a in menubar.actions()]
        assert any("文件" in t for t in actions), f"缺少文件菜单: {actions}"

    def test_menubar_export_submenu(self, qapp):
        """文件菜单下应有导出子菜单，包含三项导出操作"""
        from voltsnap.gui.main_window import MainWindow
        window = MainWindow()
        menubar = window.menuBar()
        file_action = next(a for a in menubar.actions() if "文件" in a.text())
        file_menu = file_action.menu()
        assert file_menu is not None

        export_action = next(
            (a for a in file_menu.actions() if a.menu() and "导出" in a.text()),
            None,
        )
        assert export_action is not None, "缺少导出子菜单"
        export_menu = export_action.menu()
        assert export_menu is not None
        export_texts = [a.text() for a in export_menu.actions()]
        assert len(export_texts) >= 3


class TestCircuitPanel:
    def test_load_components(self, qapp):
        from voltsnap.gui.circuit_panel import CircuitPanel
        panel = CircuitPanel()
        comps = [
            {"ref": "V1", "type": "voltage_source", "value": "5", "confidence": 0.9, "center": (100, 200)},
            {"ref": "R1", "type": "resistor", "value": "1k", "confidence": 0.85, "center": (200, 200)},
        ]
        panel.load_components(comps)
        assert panel.table_model.rowCount() == 2

    def test_get_components(self, qapp):
        from voltsnap.gui.circuit_panel import CircuitPanel
        panel = CircuitPanel()
        comps = [
            {"ref": "R1", "type": "resistor", "value": "10k", "confidence": 0.9, "center": (0, 0)},
        ]
        panel.load_components(comps)
        result = panel.get_components()
        assert len(result) == 1
        assert result[0]["ref"] == "R1"


class TestSimulationPanel:
    def test_netlist_editor(self, qapp):
        from voltsnap.gui.simulation_panel import SimulationPanel
        panel = SimulationPanel()
        netlist = "* test\nV1 N1 0 DC 5\nR1 N1 0 1k\n.op\n.end\n"
        panel.set_netlist(netlist)
        assert panel.get_netlist() == netlist

    def test_clear_results(self, qapp):
        from voltsnap.gui.simulation_panel import SimulationPanel
        panel = SimulationPanel()
        panel.clear_results()
        assert panel.voltage_table.rowCount() == 0


class TestImagePanel:
    def test_panel_creation(self, qapp):
        from voltsnap.gui.image_panel import ImagePanel
        panel = ImagePanel()
        assert panel.scene is not None
        assert panel.view is not None

    def test_class_colors_include_new_types(self, qapp):
        """新类型在 CLASS_COLORS 中有定义"""
        from voltsnap.gui.image_panel import ImagePanel
        for t in ["ground", "switch", "led", "npn_transistor", "pnp_transistor", "nmos", "pmos"]:
            assert t in ImagePanel.CLASS_COLORS, f"{t} missing from CLASS_COLORS"


class TestComponentTableModel:
    def test_type_display_include_new_types(self, qapp):
        """新类型在 TYPE_DISPLAY 中有中文名"""
        from voltsnap.gui.circuit_panel import ComponentTableModel
        for t in ["ground", "switch", "led", "npn_transistor", "pnp_transistor", "nmos", "pmos"]:
            assert t in ComponentTableModel.TYPE_DISPLAY, f"{t} missing from TYPE_DISPLAY"


class TestExportSimulationJson:
    """仿真结果 JSON 导出测试"""

    def test_export_json(self, qapp, tmp_path):
        from voltsnap.gui.main_window import MainWindow
        from voltsnap.models import SimulationResult

        window = MainWindow()
        result = SimulationResult(
            success=True,
            node_voltages={"N1": 5.0, "N2": 2.5},
            branch_currents={"V1": -0.0025},
        )
        out_path = str(tmp_path / "sim.json")
        window._export_simulation_json(out_path, result)

        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        assert data["success"] is True
        assert data["node_voltages"]["N1"] == 5.0
        assert "V1" in data["branch_currents"]


class TestExportSimulationCsv:
    """仿真结果 CSV 导出测试"""

    def test_export_csv(self, qapp, tmp_path):
        from voltsnap.gui.main_window import MainWindow
        from voltsnap.models import SimulationResult

        window = MainWindow()
        result = SimulationResult(
            success=True,
            node_voltages={"N1": 5.0, "N2": 2.5},
            branch_currents={"V1": -0.0025},
        )
        out_path = str(tmp_path / "sim.csv")
        window._export_simulation_csv(out_path, result)

        content = Path(out_path).read_text(encoding="utf-8")
        assert "类型" in content
        assert "N1" in content
        assert "5.000000" in content


class TestSelectionSync:
    """双向选择联动测试"""

    def _make_window_with_data(self, qapp):
        from voltsnap.gui.main_window import MainWindow
        window = MainWindow()
        comps = [
            {"ref": "V1", "type": "voltage_source", "value": "5", "confidence": 0.95, "center": (100, 50)},
            {"ref": "R1", "type": "resistor", "value": "1k", "confidence": 0.85, "center": (200, 200)},
        ]
        detections = [
            {"ref": "V1", "class_name": "voltage_source", "confidence": 0.95, "bbox": (80, 30, 120, 70)},
            {"ref": "R1", "class_name": "resistor", "confidence": 0.85, "bbox": (180, 180, 220, 220)},
        ]
        window.circuit_panel.load_components(comps)
        window.image_panel.overlay_detections(detections)
        return window

    def test_image_click_selects_table_row(self, qapp):
        """点击图像框应选中对应表格行"""
        window = self._make_window_with_data(qapp)
        # 模拟点击图像框
        window.image_panel.component_selected.emit("R1")
        # 表格应选中 R1
        indexes = window.circuit_panel.table_view.selectionModel().selectedRows()
        assert len(indexes) == 1
        row = indexes[0].row()
        ref = window.circuit_panel.table_model.data(
            window.circuit_panel.table_model.index(row, 0)
        )
        assert ref == "R1"

    def test_table_select_highlights_image(self, qapp):
        """选中表格行应高亮图像框"""
        window = self._make_window_with_data(qapp)
        window.circuit_panel.select_component("V1")
        assert window.image_panel._selected_ref == "V1"

    def test_status_bar_shows_component_info(self, qapp):
        """状态栏应显示所选元件信息"""
        window = self._make_window_with_data(qapp)
        window._update_status_for_component("R1")
        msg = window.status_bar.currentMessage()
        assert "R1" in msg
        assert "电阻" in msg
        assert "1k" in msg

    def test_select_none_clears_status(self, qapp):
        """取消选中应恢复状态栏默认消息"""
        window = self._make_window_with_data(qapp)
        window._update_status_for_component(None)
        msg = window.status_bar.currentMessage()
        assert "就绪" in msg

    def test_circuit_panel_select_component(self, qapp):
        """CircuitPanel.select_component 应高亮指定行"""
        from voltsnap.gui.circuit_panel import CircuitPanel
        panel = CircuitPanel()
        panel.load_components([
            {"ref": "R1", "type": "resistor", "value": "1k", "confidence": 0.9, "center": (0, 0)},
            {"ref": "R2", "type": "resistor", "value": "2k", "confidence": 0.8, "center": (0, 0)},
        ])
        panel.select_component("R2")
        indexes = panel.table_view.selectionModel().selectedRows()
        assert len(indexes) == 1
        assert panel.table_model.data(panel.table_model.index(indexes[0].row(), 0)) == "R2"

    def test_image_panel_select_component(self, qapp):
        """ImagePanel.select_component 应高亮指定检测框"""
        from voltsnap.gui.image_panel import ImagePanel
        panel = ImagePanel()
        # overlay_detections 需要 scene 中有 pixmap
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap(400, 300)
        pixmap.fill()
        panel.scene.addPixmap(pixmap)
        panel.overlay_detections([
            {"ref": "R1", "class_name": "resistor", "confidence": 0.9, "bbox": (10, 10, 100, 50)},
            {"ref": "R2", "class_name": "resistor", "confidence": 0.8, "bbox": (150, 10, 250, 50)},
        ])
        panel.select_component("R1")
        assert panel._selected_ref == "R1"
        assert "R1" in panel._rect_items
