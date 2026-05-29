"""数据生成模块测试"""
import pytest
from pathlib import Path

from voltsnap.datagen.circuit_templates import get_demo_circuits, RandomCircuitGenerator, RandomCircuitConfig
from voltsnap.datagen.schematic_renderer import SchematicRenderer
from voltsnap.config import Config


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path


@pytest.fixture
def renderer():
    return SchematicRenderer(dpi=Config.RENDER_DPI)


class TestCircuitTemplates:
    def test_demo_circuits_count(self):
        circuits = get_demo_circuits()
        assert len(circuits) == 3

    def test_demo_circuits_have_names(self):
        names = {c.name for c in get_demo_circuits()}
        assert names == {"resistor_divider", "parallel_resistors", "rc_circuit"}

    def test_demo_circuits_have_components(self):
        for circuit in get_demo_circuits():
            assert len(circuit.components) >= 2
            for comp in circuit.components:
                assert comp.ref
                assert comp.type
                assert comp.value


class TestRandomCircuitGenerator:
    """随机电路生成器测试"""

    def test_generate_all_topology_types(self):
        gen = RandomCircuitGenerator(seed=42)
        for topo in gen.TOPOLOGY_TYPES:
            circuit = gen.generate(topo)
            assert circuit.name == topo
            assert len(circuit.components) >= 2
            assert circuit.expected_netlist
            assert ".op" in circuit.expected_netlist

    def test_generate_random_topology(self):
        gen = RandomCircuitGenerator(seed=42)
        circuit = gen.generate()
        assert circuit.name in gen.TOPOLOGY_TYPES

    def test_generate_batch(self):
        gen = RandomCircuitGenerator(seed=42)
        circuits = gen.generate_batch(16)
        assert len(circuits) == 16
        names = [c.name for c in circuits]
        assert len(set(names)) == 16  # 唯一名称

    def test_seed_deterministic(self):
        gen1 = RandomCircuitGenerator(seed=123)
        gen2 = RandomCircuitGenerator(seed=123)
        c1 = gen1.generate("resistor_divider")
        c2 = gen2.generate("resistor_divider")
        assert c1.expected_netlist == c2.expected_netlist

    def test_different_seeds_differ(self):
        gen1 = RandomCircuitGenerator(seed=1)
        gen2 = RandomCircuitGenerator(seed=2)
        results = set()
        for _ in range(10):
            c1 = gen1.generate("resistor_divider")
            c2 = gen2.generate("resistor_divider")
            results.add(c1.expected_netlist)
            results.add(c2.expected_netlist)
        # 种子不同，参数应有差异
        assert len(results) > 1

    def test_voltage_source_present(self):
        gen = RandomCircuitGenerator(seed=42)
        for topo in gen.TOPOLOGY_TYPES:
            circuit = gen.generate(topo)
            sources = [c for c in circuit.components if c.type == "voltage_source"]
            assert len(sources) >= 1, f"{topo} 缺少电压源"

    def test_netlist_has_components(self):
        gen = RandomCircuitGenerator(seed=42)
        circuit = gen.generate("two_mesh")
        netlist = circuit.expected_netlist
        assert "V1" in netlist
        assert "R1" in netlist
        assert "R2" in netlist


class TestSchematicRenderer:
    def test_render_resistor_divider(self, renderer, output_dir):
        circuit = get_demo_circuits()[0]
        out = str(output_dir / "divider.png")
        result = renderer.render(circuit, out)
        assert Path(out).exists()
        assert result.image_size[0] > 0
        assert result.image_size[1] > 0

    def test_render_parallel_resistors(self, renderer, output_dir):
        circuit = get_demo_circuits()[1]
        out = str(output_dir / "parallel.png")
        result = renderer.render(circuit, out)
        assert Path(out).exists()

    def test_render_rc_circuit(self, renderer, output_dir):
        circuit = get_demo_circuits()[2]
        out = str(output_dir / "rc.png")
        result = renderer.render(circuit, out)
        assert Path(out).exists()

    def test_component_pins_extracted(self, renderer, output_dir):
        circuit = get_demo_circuits()[0]
        out = str(output_dir / "test.png")
        result = renderer.render(circuit, out)
        for comp in result.components:
            assert len(comp.pins) == 2
            for pin in comp.pins:
                assert pin.pixel_position != (0, 0)

    def test_all_demo_circuits_render(self, renderer, output_dir):
        for circuit in get_demo_circuits():
            out = str(output_dir / f"{circuit.name}.png")
            result = renderer.render(circuit, out)
            assert Path(out).exists()
            assert len(result.components) >= 2
