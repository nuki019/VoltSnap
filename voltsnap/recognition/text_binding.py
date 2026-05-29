"""文字-元件绑定 — 匈牙利算法全局最优匹配"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("voltsnap.recognition.binding")


@dataclass
class BoundComponent:
    """绑定后的元件信息"""
    ref: str                  # 元件编号 (R1, C2, ...)
    type: str                 # 元件类型
    value: str                # 参数值
    bbox: tuple[int, int, int, int]  # 元件检测框
    center: tuple[int, int]  # 元件中心
    confidence: float        # 检测置信度
    angle: float             # 旋转角度
    value_confidence: float  # 参数绑定置信度


# 类型-单位约束矩阵权重
# 行：元件类型，列：参数类型（电阻值、电容值、电感值、电压值、电流值）
_CONSTRAINT_MATRIX = {
    "resistor":       [1.0, 0.0, 0.0, 0.1, 0.1],
    "capacitor":      [0.0, 1.0, 0.0, 0.1, 0.1],
    "inductor":       [0.0, 0.0, 1.0, 0.1, 0.1],
    "voltage_source": [0.0, 0.0, 0.0, 1.0, 0.0],
    "current_source": [0.0, 0.0, 0.0, 0.0, 1.0],
}


def _infer_value_type(value_text: str) -> int:
    """从参数值推断类型索引：0=电阻 1=电容 2=电感 3=电压 4=电流"""
    v = value_text.lower()
    if any(u in v for u in ["k", "meg", "ohm"]) or (v.replace(".", "").isdigit()):
        return 0  # 电阻
    if any(u in v for u in ["uf", "nf", "pf", "f"]):
        return 1  # 电容
    if any(u in v for u in ["mh", "uh", "h"]) and "v" not in v:
        return 2  # 电感
    if any(u in v for u in ["ma", "ua"]) or (v.endswith("a") and not v.endswith("va")):
        return 4  # 电流
    if "v" in v:
        return 3  # 电压
    return 0  # 默认电阻


class TextBinder:
    """
    文字-元件绑定器。

    使用匈牙利算法将 OCR 识别的文本（编号、参数值）绑定到检测到的元件上。
    结合空间距离和电学语义约束构建代价矩阵。
    """

    def bind(
        self,
        detections: list[dict],
        ocr_results: list[dict],
        max_distance: float = 200.0,
    ) -> list[BoundComponent]:
        """
        将 OCR 文本绑定到检测到的元件。

        Parameters
        ----------
        detections : list[dict]
            元件检测结果，每个 dict 含:
            - class_name: str
            - bbox: [x1, y1, x2, y2]
            - center: [cx, cy]
            - confidence: float
            - angle: float
        ocr_results : list[dict]
            OCR 结果，每个 dict 含:
            - text: str
            - bbox: [x1, y1, x2, y2]
            - is_ref: bool
            - is_value: bool
            - normalized: str
            - component_type: str
        max_distance : float
            最大绑定距离（像素）。超过此距离的匹配会被拒绝。

        Returns
        -------
        list[BoundComponent]
            绑定后的元件列表。
        """
        if not detections:
            return []

        # 分离编号和参数值
        refs = [r for r in ocr_results if r.get("is_ref")]
        values = [r for r in ocr_results if r.get("is_value")]

        # 初始化绑定结果
        bound_list = []
        for det in detections:
            bound_list.append(BoundComponent(
                ref="",
                type=det.get("class_name", "unknown"),
                value="",
                bbox=tuple(det.get("bbox", [0, 0, 0, 0])),
                center=tuple(det.get("center", [0, 0])),
                confidence=det.get("confidence", 0.0),
                angle=det.get("angle", 0.0),
                value_confidence=0.0,
            ))

        # 绑定编号
        if refs:
            ref_matches = self._match_texts_to_detections(
                refs, detections, max_distance, is_ref=True
            )
            for det_idx, text_idx, score in ref_matches:
                if text_idx >= 0:
                    bound_list[det_idx].ref = refs[text_idx].get("normalized", "")

        # 绑定参数值
        if values:
            value_matches = self._match_texts_to_detections(
                values, detections, max_distance, is_ref=False
            )
            for det_idx, text_idx, score in value_matches:
                if text_idx >= 0:
                    bound_list[det_idx].value = values[text_idx].get("normalized", "")
                    bound_list[det_idx].value_confidence = score

        return bound_list

    def _match_texts_to_detections(
        self,
        texts: list[dict],
        detections: list[dict],
        max_distance: float,
        is_ref: bool,
    ) -> list[tuple[int, int, float]]:
        """
        使用匈牙利算法匹配文本到元件。

        Returns
        -------
        list[tuple[int, int, float]]
            (检测索引, 文本索引, 匹配分数) 列表。
        """
        n_det = len(detections)
        n_text = len(texts)

        # 构建代价矩阵
        cost = np.full((n_det, n_text), 1e6)

        for i, det in enumerate(detections):
            det_center = np.array(det.get("center", [0, 0]))
            det_type = det.get("class_name", "unknown")

            for j, text in enumerate(texts):
                text_center = np.array(self._bbox_center(text.get("bbox", [0, 0, 0, 0])))

                # 空间距离代价
                dist = np.linalg.norm(det_center - text_center)
                if dist > max_distance:
                    continue

                dist_cost = dist / max_distance  # 归一化到 [0, 1]

                # 语义约束
                if is_ref:
                    # 编号匹配：类型前缀必须一致
                    text_type = text.get("component_type", "")
                    semantic_bonus = 0.3 if text_type == det_type else 0.0
                else:
                    # 参数值匹配：类型-单位约束
                    value_type_idx = _infer_value_type(text.get("normalized", ""))
                    constraint = _CONSTRAINT_MATRIX.get(det_type, [0.5] * 5)
                    semantic_bonus = (1.0 - constraint[value_type_idx]) * 0.5

                cost[i, j] = dist_cost + semantic_bonus

        # 匈牙利算法
        matches = []
        if n_det > 0 and n_text > 0:
            try:
                from scipy.optimize import linear_sum_assignment
                row_indices, col_indices = linear_sum_assignment(cost)

                for r, c in zip(row_indices, col_indices):
                    if cost[r, c] < 1.0:  # 代价 < 1.0 表示有效匹配
                        score = 1.0 - cost[r, c]
                        matches.append((r, c, max(0.0, score)))
                    else:
                        matches.append((r, -1, 0.0))
            except ImportError:
                # 回退：简单最近邻
                matches = self._greedy_match(cost, n_det)

        return matches

    @staticmethod
    def _greedy_match(cost: np.ndarray, n_det: int) -> list[tuple[int, int, float]]:
        """贪心最近邻匹配（当 scipy 不可用时的回退方案）"""
        matches = []
        used_texts = set()

        for i in range(n_det):
            if cost.shape[1] == 0:
                matches.append((i, -1, 0.0))
                continue

            row = cost[i]
            best_j = -1
            best_cost = 1e6

            for j in range(len(row)):
                if j not in used_texts and row[j] < best_cost:
                    best_cost = row[j]
                    best_j = j

            if best_cost < 1.0 and best_j >= 0:
                used_texts.add(best_j)
                matches.append((i, best_j, max(0.0, 1.0 - best_cost)))
            else:
                matches.append((i, -1, 0.0))

        return matches

    @staticmethod
    def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        """计算 bbox 中心"""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
