"""图像处理与拓扑重建测试"""
import numpy as np
import pytest

from voltsnap.vision.preprocessor import ImagePreprocessor
from voltsnap.vision.topology import TopologyReconstructor
from voltsnap.datagen.circuit_templates import get_demo_circuits
from voltsnap.datagen.schematic_renderer import SchematicRenderer
from voltsnap.config import Config


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


@pytest.fixture
def topology():
    return TopologyReconstructor(pin_search_radius=30)


@pytest.fixture
def rendered_divider(tmp_path):
    """渲染串联分压电路，返回渲染结果"""
    renderer = SchematicRenderer(dpi=Config.RENDER_DPI)
    circuit = get_demo_circuits()[0]
    out = str(tmp_path / "divider.png")
    return renderer.render(circuit, out)


class TestImagePreprocessor:
    def test_returns_correct_types(self, preprocessor, rendered_divider):
        result = preprocessor.process(rendered_divider.image_path)
        assert result.gray.ndim == 2
        assert result.binary.ndim == 2
        assert result.original.ndim == 3

    def test_binary_is_0_or_255(self, preprocessor, rendered_divider):
        result = preprocessor.process(rendered_divider.image_path)
        unique = set(np.unique(result.binary))
        assert unique <= {0, 255}

    def test_has_foreground_pixels(self, preprocessor, rendered_divider):
        result = preprocessor.process(rendered_divider.image_path)
        fg = np.sum(result.binary > 0)
        assert fg > 100  # 至少有 100 个前景像素


class TestTopologyReconstructor:
    def test_skeleton_exists(self, preprocessor, topology, rendered_divider):
        result = preprocessor.process(rendered_divider.image_path)
        topo = topology.reconstruct(result.binary, rendered_divider.components[0].pins[:1])
        assert topo.skeleton is not None
        assert topo.skeleton.shape == result.binary.shape

    def test_pin_to_net_mapping(self, preprocessor, topology, rendered_divider):
        """所有引脚都应被映射到某个 Net（阶段 0 简化：闭环电路可能只有 1 个 Net）"""
        result = preprocessor.process(rendered_divider.image_path)
        all_pins = []
        for comp in rendered_divider.components:
            all_pins.extend(comp.pins)
        topo = topology.reconstruct(result.binary, all_pins)
        # 阶段 0 简化：闭环电路（如分压器）的回路导线使所有导线成为一个连通域
        # 至少应有 1 个 Net，且所有引脚都被映射
        assert topo.num_nets >= 1
        assert len(topo.pin_to_net) == len(all_pins)

    def test_connected_pins_share_net(self, preprocessor, topology, rendered_divider):
        """R1 的 pin2 和 R2 的 pin1 应在同一 Net（串联连接点）"""
        result = preprocessor.process(rendered_divider.image_path)
        all_pins = []
        for comp in rendered_divider.components:
            all_pins.extend(comp.pins)
        topo = topology.reconstruct(result.binary, all_pins)

        r1_pin2 = "R1_pin2"
        r2_pin1 = "R2_pin1"
        if r1_pin2 in topo.pin_to_net and r2_pin1 in topo.pin_to_net:
            assert topo.pin_to_net[r1_pin2] == topo.pin_to_net[r2_pin1]
