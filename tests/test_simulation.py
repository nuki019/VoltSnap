"""仿真模块测试"""
import pytest

from voltsnap.models import ComponentInfo, PinInfo
from voltsnap.datagen.netlist_generator import NetlistGenerator
from voltsnap.simulation.ngspice_runner import NgspiceRunner
from voltsnap.simulation.netlist_sanitizer import NetlistSanitizer
from voltsnap.config import Config


@pytest.fixture
def netlist_gen():
    return NetlistGenerator()


@pytest.fixture
def sanitizer():
    return NetlistSanitizer()


class TestNetlistGenerator:
    def test_resistor_divider_format(self, netlist_gen):
        comps = [
            ComponentInfo("V1", "voltage_source", "5", [
                PinInfo("V1_pin1", "V1", (0, 0)),
                PinInfo("V1_pin2", "V1", (0, 0)),
            ]),
            ComponentInfo("R1", "resistor", "1k", [
                PinInfo("R1_pin1", "R1", (0, 0)),
                PinInfo("R1_pin2", "R1", (0, 0)),
            ]),
            ComponentInfo("R2", "resistor", "2k", [
                PinInfo("R2_pin1", "R2", (0, 0)),
                PinInfo("R2_pin2", "R2", (0, 0)),
            ]),
        ]
        pin_to_net = {
            "V1_pin1": 1, "V1_pin2": 3,
            "R1_pin1": 1, "R1_pin2": 2,
            "R2_pin1": 2, "R2_pin2": 3,
        }
        netlist = netlist_gen.generate(comps, pin_to_net, ground_net_id=3)
        assert ".op" in netlist
        assert ".end" in netlist
        assert " 0 " in netlist  # GND 节点

    def test_has_ground_node(self, netlist_gen):
        comps = [
            ComponentInfo("V1", "voltage_source", "5", [
                PinInfo("V1_pin1", "V1", (0, 0)),
                PinInfo("V1_pin2", "V1", (0, 0)),
            ]),
        ]
        pin_to_net = {"V1_pin1": 1, "V1_pin2": 2}
        netlist = netlist_gen.generate(comps, pin_to_net, ground_net_id=2)
        assert " 0 " in netlist


class TestNetlistSanitizer:
    def test_detect_dangling_pin(self, sanitizer):
        pin_to_net = {"R1_pin1": 1, "R1_pin2": 2}
        net_pins = {1: ["R1_pin1"], 2: ["R1_pin2"]}
        dangling = sanitizer.check_dangling_pins(pin_to_net, net_pins)
        assert len(dangling) == 2

    def test_no_dangling_pins(self, sanitizer):
        pin_to_net = {"R1_pin1": 1, "R1_pin2": 2, "R2_pin1": 1, "R2_pin2": 2}
        net_pins = {1: ["R1_pin1", "R2_pin1"], 2: ["R1_pin2", "R2_pin2"]}
        dangling = sanitizer.check_dangling_pins(pin_to_net, net_pins)
        assert len(dangling) == 0


class TestNgspiceRunner:
    """注意：这些测试需要 ngspice 已安装"""

    def test_resistor_divider_voltage(self):
        netlist = (
            "* Test\n"
            "V1 N1 0 DC 5\n"
            "R1 N1 N2 1k\n"
            "R2 N2 0 2k\n"
            ".op\n"
            ".end\n"
        )
        runner = NgspiceRunner(ngspice_path=Config.NGSPICE_PATH)
        result = runner.run(netlist)
        if not result.success:
            pytest.skip(f"ngspice not available: {result.error_message}")
        # 分压：V(N2) = 5 * 2k/(1k+2k) = 3.333V
        assert "N2" in result.node_voltages
        assert abs(result.node_voltages["N2"] - 3.333) < 0.05

    def test_invalid_netlist_returns_error(self):
        netlist = "* Bad\n.short V1 0\n.end\n"
        runner = NgspiceRunner(ngspice_path=Config.NGSPICE_PATH)
        result = runner.run(netlist)
        # 应该失败或返回异常值
        assert not result.success or result.error_message is not None
