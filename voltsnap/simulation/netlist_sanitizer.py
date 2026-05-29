"""网表净化：GND 判定、悬空引脚修复"""
from __future__ import annotations

import logging
from typing import List

from voltsnap.models import ComponentInfo

logger = logging.getLogger("voltsnap.simulation.sanitizer")


class NetlistSanitizer:
    """对网表生成前的拓扑进行静态检查和修复"""

    def __init__(self, high_resistance: str = "1e12"):
        self.high_resistance = high_resistance

    def check_dangling_pins(
        self,
        pin_to_net: dict[str, int],
        net_pins: dict[int, list[str]],
    ) -> list[str]:
        """
        检测悬空引脚：某个 Net 只连接了单个元件的单个引脚。

        Returns:
            悬空引脚名列表
        """
        dangling: list[str] = []
        for net_id, pins in net_pins.items():
            # 检查是否只有一个引脚
            if len(pins) == 1:
                dangling.append(pins[0])
                logger.warning("Dangling pin detected: %s on net %d", pins[0], net_id)
        return dangling

    def has_ground(self, pin_to_net: dict[str, int], ground_net: int) -> bool:
        """检查是否存在 GND 节点"""
        return ground_net in pin_to_net.values()
