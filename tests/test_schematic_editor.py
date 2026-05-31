"""SchematicEditor 测试（offscreen 模式）"""
import sys

import pytest

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ── 测试数据 ──────────────────────────────────────────────────────────────

def _sample_components():
    """返回一份新的测试元件列表（避免测试间共享可变状态）"""
    return [
        {"ref": "V1", "type": "voltage_source", "value": "5V", "confidence": 0.95, "center": (50, 100)},
        {"ref": "R1", "type": "resistor", "value": "1k", "confidence": 0.90, "center": (200, 100)},
        {"ref": "R2", "type": "resistor", "value": "2k", "confidence": 0.85, "center": (350, 100)},
        {"ref": "C1", "type": "capacitor", "value": "10uF", "confidence": 0.80, "center": (200, 250)},
        {"ref": "L1", "type": "inductor", "value": "10mH", "confidence": 0.75, "center": (350, 250)},
        {"ref": "D1", "type": "diode", "value": "1N4148", "confidence": 0.70, "center": (500, 100)},
        {"ref": "GND", "type": "ground", "value": "", "confidence": 0.99, "center": (500, 250)},
    ]


# ── 基础功能 ──────────────────────────────────────────────────────────────

class TestSchematicEditorCreation:
    def test_widget_creation(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        assert editor.scene is not None
        assert editor.view is not None
        assert editor.title_label is not None

    def test_empty_load(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components([])
        assert len(editor._comp_items) == 0
        assert "无元件" in editor.title_label.text()


class TestSchematicEditorLoadComponents:
    def test_load_creates_items(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        comps = _sample_components()
        editor.load_components(comps)
        assert len(editor._comp_items) == len(comps)

    def test_all_refs_present(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        comps = _sample_components()
        editor.load_components(comps)
        for comp in comps:
            assert comp["ref"] in editor._comp_items

    def test_reload_clears_old(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        comps = _sample_components()
        editor.load_components(comps)
        editor.load_components([comps[0]])
        assert len(editor._comp_items) == 1
        assert "V1" in editor._comp_items

    def test_get_components(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        comps = _sample_components()
        editor.load_components(comps)
        result = editor.get_components()
        assert len(result) == len(comps)

    def test_reload_components_alias(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        comps = _sample_components()
        editor.reload_components(comps)
        assert len(editor._comp_items) == len(comps)

    def test_title_shows_count(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        comps = _sample_components()
        editor.load_components(comps)
        assert str(len(comps)) in editor.title_label.text()


# ── 选择功能 ──────────────────────────────────────────────────────────────

class TestSchematicEditorSelection:
    def test_select_component_highlights(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.select_component("R1")
        assert editor._selected_ref == "R1"
        assert editor._comp_items["R1"]._selected is True

    def test_select_none_clears(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.select_component("R1")
        editor.select_component(None)
        assert editor._selected_ref is None

    def test_select_switches_highlight(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.select_component("R1")
        editor.select_component("R2")
        assert editor._comp_items["R1"]._selected is False
        assert editor._comp_items["R2"]._selected is True

    def test_click_emits_signal(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())

        received = []
        editor.component_selected.connect(lambda ref: received.append(ref))
        editor._on_component_clicked("C1")
        assert received == ["C1"]

    def test_select_nonexistent_ref_no_crash(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.select_component("ZX")  # 不应崩溃


# ── 刷新功能 ──────────────────────────────────────────────────────────────

class TestSchematicEditorRefresh:
    def test_refresh_component_value(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.refresh_component("R1", {"value": "4.7k"})
        assert "R1" in editor._comp_items
        comp = next(c for c in editor._components if c["ref"] == "R1")
        assert comp["value"] == "4.7k"

    def test_refresh_component_ref(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.select_component("R1")
        editor.refresh_component("R1", {"ref": "R1_new", "value": "10k"})
        assert "R1_new" in editor._comp_items
        assert "R1" not in editor._comp_items
        assert editor._selected_ref == "R1_new"

    def test_refresh_nonexistent_no_crash(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.refresh_component("NOPE", {"value": "x"})  # 不应崩溃

    def test_component_items_are_movable(self, qapp):
        from PyQt6.QtWidgets import QGraphicsItem
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(_sample_components())
        flags = editor._comp_items["R1"].flags()
        assert flags & QGraphicsItem.GraphicsItemFlag.ItemIsMovable


# ── 导线和画布编辑 ──────────────────────────────────────────────────────────

class TestSchematicEditorWires:
    def test_load_components_auto_generates_wires(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(_sample_components())
        assert len(editor._wire_items) > 0

    def test_load_components_uses_detected_connections(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(
            _sample_components(),
            connections=[
                {
                    "start_ref": "V1",
                    "end_ref": "R1",
                    "start_pin": "top",
                    "end_pin": "left",
                },
                {
                    "start_ref": "R1",
                    "end_ref": "R2",
                    "start_pin": "right",
                    "end_pin": "left",
                },
            ],
        )

        assert len(editor._wire_items) == 2
        assert {(w.start_ref, w.end_ref, w.start_pin, w.end_pin) for w in editor._wire_items} == {
            ("V1", "R1", "top", "left"),
            ("R1", "R2", "right", "left"),
        }

    def test_parallel_wires_between_same_components_keep_pin_sides(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.clear_wires()

        assert editor.add_wire_between_refs("V1", "R1", "top", "left") is not None
        assert editor.add_wire_between_refs("V1", "R1", "bottom", "right") is not None

        assert len(editor._wire_items) == 2

    def test_add_select_delete_wire(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.clear_wires()
        wire = editor.add_wire_between_refs("V1", "R1")
        assert wire is not None
        assert wire in editor._wire_items

        editor.select_wire(wire)
        assert editor._selected_wire is wire
        assert wire._selected is True
        assert editor.delete_selected_wire() is True
        assert wire not in editor._wire_items
        assert editor._selected_wire is None

    def test_wire_updates_when_component_moves(self, qapp):
        from PyQt6.QtCore import QPointF
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.clear_wires()
        wire = editor.add_wire_between_refs("V1", "R1")
        before = wire.path().boundingRect()
        editor._comp_items["R1"].setPos(editor._comp_items["R1"].pos() + QPointF(120, 0))
        after = wire.path().boundingRect()
        assert after != before

    def test_refresh_ref_updates_wire_endpoints(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(_sample_components())
        editor.clear_wires()
        wire = editor.add_wire_between_refs("V1", "R1")
        editor.refresh_component("R1", {"ref": "R10"})
        assert wire in editor._wire_items
        assert wire.end_ref == "R10"

    def test_canvas_component_update_emits_and_refreshes(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor

        editor = SchematicEditor()
        editor.load_components(_sample_components())
        received = []
        editor.component_changed.connect(lambda old_ref, updates: received.append((old_ref, updates)))

        assert editor.update_component_from_canvas(
            "R1",
            {"ref": "R10", "type": "capacitor", "value": "22uF"},
        )
        assert "R10" in editor._comp_items
        assert "R1" not in editor._comp_items
        assert editor._comp_items["R10"].comp_type == "capacitor"
        assert editor._comp_items["R10"].value == "22uF"
        assert received == [("R1", {"ref": "R10", "type": "capacitor", "value": "22uF"})]


# ── 仿真叠加 ──────────────────────────────────────────────────────────────

class _FakeSimResult:
    """模拟 SimulationResult"""
    def __init__(self, node_voltages, branch_currents):
        self.node_voltages = node_voltages
        self.branch_currents = branch_currents


class TestSchematicEditorSimulationOverlay:
    def test_show_simulation_labels(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        result = _FakeSimResult(
            node_voltages={"v(n1)": 5.0, "v(n2)": 2.5},
            branch_currents={"i(v1)": -0.0025, "i(r1)": 0.0025},
        )
        editor.show_simulation_result(result)
        r1_item = editor._comp_items["R1"]
        assert len(r1_item._sim_texts) > 0
        assert len(editor._overlay_items) > 0

    def test_clear_simulation_overlay(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        result = _FakeSimResult(
            node_voltages={"v(n1)": 5.0},
            branch_currents={"i(r1)": 0.001},
        )
        editor.show_simulation_result(result)
        editor.clear_simulation_overlay()
        for item in editor._comp_items.values():
            assert len(item._sim_texts) == 0
        assert len(editor._overlay_items) == 0

    def test_unmatched_node_voltages_show_summary(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        editor.load_components(_sample_components())
        result = _FakeSimResult(
            node_voltages={"N1": 5.0, "N2": 2.5},
            branch_currents={"V1#branch": -0.0025},
        )
        editor.show_simulation_result(result)
        assert len(editor._overlay_items) == 2

    def test_show_then_reload_clears(self, qapp):
        from voltsnap.gui.schematic_editor import SchematicEditor
        editor = SchematicEditor()
        comps = _sample_components()
        editor.load_components(comps)
        result = _FakeSimResult(
            node_voltages={"v(n1)": 5.0},
            branch_currents={"i(r1)": 0.001},
        )
        editor.show_simulation_result(result)
        editor.load_components([comps[0]])
        assert len(editor._comp_items) == 1


# ── ComponentItem 符号类型 ────────────────────────────────────────────────

class TestComponentItemTypes:
    """验证各类型元件都能正常创建"""

    TYPES = [
        "resistor", "capacitor", "inductor",
        "voltage_source", "current_source", "ground",
        "diode", "led", "op_amp", "switch",
        "npn_transistor", "pnp_transistor", "nmos", "pmos", "unknown",
    ]

    @pytest.mark.parametrize("comp_type", TYPES)
    def test_create_each_type(self, qapp, comp_type):
        from voltsnap.gui.schematic_editor import ComponentItem
        comp = {"ref": "X1", "type": comp_type, "value": "1k", "confidence": 1.0, "center": (0, 0)}
        item = ComponentItem(comp)
        assert item.ref == "X1"
        assert item.comp_type == comp_type
        assert len(item._body_items) > 0
