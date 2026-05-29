"""从元件列表和拓扑关系生成 SPICE 网表"""
from __future__ import annotations

import logging
from typing import List

from voltsnap.models import ComponentInfo, SimulationResult

logger = logging.getLogger("voltsnap.datagen.netlist")


class NetlistGenerator:
    """从元件信息和引脚-Net 映射生成 SPICE 网表字符串"""

    def generate(
        self,
        components: List[ComponentInfo],
        pin_to_net: dict[str, int],
        ground_net_id: int | None = None,
    ) -> str:
        """
        生成 SPICE 网表。

        Args:
            components: 元件列表
            pin_to_net: 引脚名到 Net ID 的映射
            ground_net_id: 指定为 GND 的 Net ID（默认取连接元件最多的）
        """
        # 确定 GND Net
        if ground_net_id is None:
            ground_net_id = self._auto_detect_ground(pin_to_net, components)

        # 建立 Net ID 到 SPICE 节点名的映射
        net_to_node: dict[int, str] = {}
        for net_id in set(pin_to_net.values()):
            if net_id == ground_net_id:
                net_to_node[net_id] = "0"
            else:
                net_to_node[net_id] = f"N{net_id}"

        lines = ["* VoltSnap Auto-Generated Netlist"]

        for comp in components:
            if len(comp.pins) < 2:
                continue
            pin1_name = comp.pins[0].name
            pin2_name = comp.pins[1].name
            node1 = net_to_node.get(pin_to_net.get(pin1_name, -1), "NC")
            node2 = net_to_node.get(pin_to_net.get(pin2_name, -1), "NC")

            line = self._format_component(comp, node1, node2)
            if line:
                lines.append(line)

        lines.append(".op")
        lines.append(".end")

        netlist = "\n".join(lines) + "\n"
        logger.info("Generated netlist:\n%s", netlist)
        return netlist

    def _format_component(self, comp: ComponentInfo, node1: str, node2: str) -> str | None:
        ref = comp.ref
        typ = comp.type
        val = comp.value

        if typ == "resistor":
            return f"R{ref[1:]} {node1} {node2} {val}"
        elif typ == "capacitor":
            return f"C{ref[1:]} {node1} {node2} {val}"
        elif typ == "voltage_source":
            return f"V{ref[1:]} {node1} {node2} DC {val}"
        elif typ == "current_source":
            return f"I{ref[1:]} {node1} {node2} DC {val}"
        else:
            logger.warning("Unknown component type: %s", typ)
            return None

    def _auto_detect_ground(
        self, pin_to_net: dict[str, int], components: List[ComponentInfo]
    ) -> int:
        """选择连接引脚数最多的 Net 作为 GND"""
        from collections import Counter

        net_counts: Counter[int] = Counter()
        for net_id in pin_to_net.values():
            net_counts[net_id] += 1

        if not net_counts:
            return 0

        return net_counts.most_common(1)[0][0]
