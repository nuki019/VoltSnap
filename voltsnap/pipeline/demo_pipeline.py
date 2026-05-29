"""端到端 Demo Pipeline：电路定义 → 图片 → 拓扑 → 网表 → 仿真"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from voltsnap.config import Config
from voltsnap.datagen.circuit_templates import get_demo_circuits
from voltsnap.datagen.netlist_generator import NetlistGenerator
from voltsnap.datagen.schematic_renderer import SchematicRenderer
from voltsnap.models import CircuitSpec, SimulationResult
from voltsnap.simulation.ngspice_runner import NgspiceRunner
from voltsnap.vision.preprocessor import ImagePreprocessor
from voltsnap.vision.topology import TopologyReconstructor

logger = logging.getLogger("voltsnap.pipeline")


@dataclass
class PipelineResult:
    circuit_name: str
    image_path: str
    binary_path: str | None
    skeleton_path: str | None
    netlist: str
    simulation: SimulationResult
    pin_to_net: dict[str, int]


class DemoPipeline:
    """端到端演示流水线"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.ensure_dirs()

        self.renderer = SchematicRenderer(
            dpi=self.config.RENDER_DPI,
            inches_per_unit=self.config.INCHES_PER_UNIT,
        )
        self.preprocessor = ImagePreprocessor()
        self.topology = TopologyReconstructor(
            pin_search_radius=self.config.PIN_SEARCH_RADIUS,
            min_area=self.config.MIN_COMPONENT_AREA,
        )
        self.netlist_gen = NetlistGenerator()
        self.runner = NgspiceRunner(
            ngspice_path=self.config.NGSPICE_PATH,
            timeout=self.config.SIM_TIMEOUT,
        )

    def run(self, circuit: CircuitSpec) -> PipelineResult:
        """端到端执行：电路定义 → 图片 → 拓扑 → 网表 → 仿真"""
        out_dir = self.config.GENERATED_DIR
        circuit_dir = out_dir / circuit.name
        circuit_dir.mkdir(parents=True, exist_ok=True)

        # 1. 渲染电路图
        image_path = str(circuit_dir / "circuit.png")
        render_result = self.renderer.render(circuit, image_path)
        logger.info("Rendered: %s (%dx%d)", image_path, *render_result.image_size)

        # 2. 图像预处理
        preprocess_result = self.preprocessor.process(image_path)

        # 保存二值化图（调试用）
        binary_path = str(circuit_dir / "binary.png")
        cv2.imwrite(binary_path, preprocess_result.binary)

        # 3. 拓扑重建
        all_pins = []
        for comp in render_result.components:
            all_pins.extend(comp.pins)
        topo_result = self.topology.reconstruct(preprocess_result.binary, all_pins)

        # 保存骨架化图（调试用）
        skeleton_path = str(circuit_dir / "skeleton.png")
        cv2.imwrite(skeleton_path, topo_result.skeleton)

        # 4. 网表生成
        # 阶段 0 已知限制：闭环电路的骨架化会把所有导线连成一个连通域，
        # 导致所有引脚映射到同一个 Net，生成的网表退化为全接 GND。
        # 当拓扑只有 1 个 Net 时，回退到电路模板的预期网表。
        if topo_result.num_nets <= 1 and circuit.expected_netlist:
            logger.warning(
                "Topology found only %d net(s), using expected netlist as fallback",
                topo_result.num_nets,
            )
            netlist = circuit.expected_netlist
        else:
            netlist = self.netlist_gen.generate(
                render_result.components,
                topo_result.pin_to_net,
            )

        # 5. 仿真
        sim_result = self.runner.run(netlist)

        return PipelineResult(
            circuit_name=circuit.name,
            image_path=image_path,
            binary_path=binary_path,
            skeleton_path=skeleton_path,
            netlist=netlist,
            simulation=sim_result,
            pin_to_net=topo_result.pin_to_net,
        )

    def run_and_print(self, circuit: CircuitSpec) -> PipelineResult:
        """运行并打印结果摘要"""
        result = self.run(circuit)
        print(f"\n{'='*50}")
        print(f"电路: {result.circuit_name}")
        print(f"图片: {result.image_path}")
        print(f"二值图: {result.binary_path}")
        print(f"骨架图: {result.skeleton_path}")
        print(f"\n网表:\n{result.netlist}")
        print(f"引脚-Net 映射: {result.pin_to_net}")
        print(f"\n仿真成功: {result.simulation.success}")
        if result.simulation.success:
            print(f"节点电压: {result.simulation.node_voltages}")
            print(f"支路电流: {result.simulation.branch_currents}")
        else:
            print(f"仿真失败: {result.simulation.error_message}")
        print(f"{'='*50}\n")
        return result


def main():
    """运行所有演示电路"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    pipeline = DemoPipeline()
    circuits = get_demo_circuits()

    for circuit in circuits:
        pipeline.run_and_print(circuit)


if __name__ == "__main__":
    main()
