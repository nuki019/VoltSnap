"""GUI 模块测试（无需显示器）"""
import sys
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
