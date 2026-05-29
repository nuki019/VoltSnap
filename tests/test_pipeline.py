"""端到端 Pipeline 测试"""
import pytest

from voltsnap.pipeline.demo_pipeline import DemoPipeline
from voltsnap.datagen.circuit_templates import get_demo_circuits
from voltsnap.config import Config


@pytest.fixture
def pipeline():
    return DemoPipeline()


class TestDemoPipeline:
    def test_resistor_divider_e2e(self, pipeline):
        """串联分压：全流程应成功，V(N2) ≈ 3.33V"""
        circuit = get_demo_circuits()[0]
        result = pipeline.run(circuit)
        assert result.circuit_name == "resistor_divider"
        assert result.netlist
        if result.simulation.success:
            assert abs(result.simulation.node_voltages.get("N2", 0) - 3.333) < 0.05

    def test_parallel_resistors_e2e(self, pipeline):
        """并联电阻：全流程应成功"""
        circuit = get_demo_circuits()[1]
        result = pipeline.run(circuit)
        assert result.circuit_name == "parallel_resistors"
        assert result.netlist

    def test_rc_circuit_e2e(self, pipeline):
        """RC 电路：全流程应成功"""
        circuit = get_demo_circuits()[2]
        result = pipeline.run(circuit)
        assert result.circuit_name == "rc_circuit"
        assert result.netlist

    def test_debug_images_saved(self, pipeline):
        """调试图片应被保存"""
        circuit = get_demo_circuits()[0]
        result = pipeline.run(circuit)
        assert result.binary_path is not None
        assert result.skeleton_path is not None
