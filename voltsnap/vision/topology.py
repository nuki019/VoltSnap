"""拓扑重建：骨架化 + 连通域 + 引脚吸附"""
from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from voltsnap.models import PinInfo, TopologyResult

logger = logging.getLogger("voltsnap.vision.topology")


class TopologyReconstructor:
    """从二值化图和引脚坐标重建电气拓扑"""

    def __init__(self, pin_search_radius: int = 15, min_area: int = 50):
        """
        Args:
            pin_search_radius: 引脚吸附搜索半径（像素）
            min_area: 过滤噪点的最小连通域面积
        """
        self.pin_search_radius = pin_search_radius
        self.min_area = min_area

    def reconstruct(
        self,
        binary_image: np.ndarray,
        pin_positions: List[PinInfo],
    ) -> TopologyResult:
        """
        从二值图和引脚坐标重建拓扑。

        流程：骨架化 → 连通域标记 → 引脚吸附
        """
        # 1. 骨架化
        skeleton = cv2.ximgproc.thinning(
            binary_image, cv2.ximgproc.THINNING_ZHANGSUEN
        )
        logger.info("Skeletonized: foreground=%d", int(np.sum(skeleton > 0)))

        # 2. 连通域标记
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            skeleton, connectivity=8
        )
        # 过滤面积过小的连通域（噪点）
        valid_labels = set()
        for i in range(1, num_labels):  # 跳过背景 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                valid_labels.add(i)

        logger.info(
            "Connected components: total=%d, valid=%d", num_labels - 1, len(valid_labels)
        )

        # 3. 引脚吸附
        pin_to_net: dict[str, int] = {}
        net_pins: dict[int, list[str]] = {}

        for pin in pin_positions:
            px, py = pin.pixel_position
            net_id = self._find_net_for_pin(skeleton, labels, px, py)
            if net_id is not None and net_id in valid_labels:
                pin_to_net[pin.name] = net_id
                net_pins.setdefault(net_id, []).append(pin.name)
            else:
                logger.warning("Pin %s at (%d,%d) not attached to any net", pin.name, px, py)

        # 重新编号 Net ID 为连续整数（从 1 开始）
        old_to_new: dict[int, int] = {}
        new_id = 1
        for old_id in sorted(net_pins.keys()):
            old_to_new[old_id] = new_id
            new_id += 1

        # 更新映射
        remapped_pin_to_net = {
            name: old_to_new[nid] for name, nid in pin_to_net.items() if nid in old_to_new
        }
        remapped_net_pins: dict[int, list[str]] = {}
        for old_id, pins in net_pins.items():
            if old_id in old_to_new:
                remapped_net_pins[old_to_new[old_id]] = pins

        logger.info("Nets found: %d", len(remapped_net_pins))

        return TopologyResult(
            skeleton=skeleton,
            labels=labels,
            num_nets=len(remapped_net_pins),
            pin_to_net=remapped_pin_to_net,
            net_pins=remapped_net_pins,
        )

    def _find_net_for_pin(
        self,
        skeleton: np.ndarray,
        labels: np.ndarray,
        pin_x: int,
        pin_y: int,
    ) -> int | None:
        """
        在引脚坐标周围搜索最近的骨架像素，返回其连通域标签。
        """
        h, w = skeleton.shape
        radius = self.pin_search_radius
        best_dist_sq = float("inf")
        best_label = None

        y_min = max(0, pin_y - radius)
        y_max = min(h, pin_y + radius + 1)
        x_min = max(0, pin_x - radius)
        x_max = min(w, pin_x + radius + 1)

        # 提取搜索窗口
        skel_window = skeleton[y_min:y_max, x_min:x_max]
        label_window = labels[y_min:y_max, x_min:x_max]

        # 找到非零像素
        ys, xs = np.where(skel_window > 0)
        if len(ys) == 0:
            return None

        # 计算欧氏距离平方
        dy = ys.astype(np.float64) - (pin_y - y_min)
        dx = xs.astype(np.float64) - (pin_x - x_min)
        dist_sq = dx * dx + dy * dy

        min_idx = np.argmin(dist_sq)
        if dist_sq[min_idx] < radius * radius:
            best_label = int(label_window[ys[min_idx], xs[min_idx]])

        return best_label if best_label and best_label > 0 else None
