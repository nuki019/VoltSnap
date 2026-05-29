"""标注格式转换器 — 将阶段 1 标注转为 YOLO OBB 训练格式"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("voltsnap.recognition.converter")

# 元件类型到类别 ID 的映射
COMPONENT_CLASSES = {
    "resistor": 0,
    "capacitor": 1,
    "inductor": 2,
    "voltage_source": 3,
    "current_source": 4,
}


@dataclass
class OBBDetection:
    """单个 OBB 检测结果"""
    class_id: int
    cx: float      # 中心 x（归一化 0-1）
    cy: float      # 中心 y（归一化 0-1）
    w: float       # 宽度（归一化）
    h: float       # 高度（归一化）
    angle: float   # 角度（度）
    # 四角坐标（归一化），YOLO OBB 格式
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    x3: float = 0.0
    y3: float = 0.0
    x4: float = 0.0
    y4: float = 0.0


class AnnotationConverter:
    """
    将 VoltSnap 阶段 1 标注转为 YOLO OBB 训练格式。

    输入: annotation.json（含 pin_positions, image_size, components）
    输出: YOLO OBB txt 文件（每行: class_id x1 y1 x2 y2 x3 y3 x4 y4，归一化坐标）
    """

    def __init__(self, class_map: dict[str, int] | None = None):
        self.class_map = class_map or COMPONENT_CLASSES

    def convert_sample(self, annotation: dict) -> list[OBBDetection]:
        """
        将单个样本标注转为 OBB 检测列表。

        通过引脚坐标推断元件的旋转边界框：
        - 二端元件：两个引脚连线为长轴，垂直方向为短轴
        """
        img_w, img_h = annotation["image_size"]
        detections: list[OBBDetection] = []

        for comp in annotation["components"]:
            comp_ref = comp["ref"]
            comp_type = comp["type"]
            class_id = self.class_map.get(comp_type)
            if class_id is None:
                continue

            # 查找该元件的引脚
            pin_names = [k for k in annotation["pin_positions"] if k.startswith(comp_ref)]
            if len(pin_names) < 2:
                continue

            pins = [annotation["pin_positions"][pn] for pn in sorted(pin_names)]
            p1 = np.array(pins[0])
            p2 = np.array(pins[1])

            # 计算 OBB
            center = (p1 + p2) / 2
            direction = p2 - p1
            length = np.linalg.norm(direction)

            if length < 5:  # 引脚重合或过近，跳过退化框
                continue

            # 元件主体长度 = 引脚间距，宽度 = 长度的 30%（经验值）
            obb_w = length
            obb_h = length * 0.3
            angle = np.degrees(np.arctan2(direction[1], direction[0]))

            # 计算四角坐标
            corners = self._compute_corners(center, obb_w, obb_h, angle)

            # 归一化
            norm_corners = []
            for cx_pt, cy_pt in corners:
                norm_corners.append((
                    np.clip(cx_pt / img_w, 0, 1),
                    np.clip(cy_pt / img_h, 0, 1),
                ))

            detections.append(OBBDetection(
                class_id=class_id,
                cx=center[0] / img_w,
                cy=center[1] / img_h,
                w=obb_w / img_w,
                h=obb_h / img_h,
                angle=angle,
                x1=norm_corners[0][0], y1=norm_corners[0][1],
                x2=norm_corners[1][0], y2=norm_corners[1][1],
                x3=norm_corners[2][0], y3=norm_corners[2][1],
                x4=norm_corners[3][0], y4=norm_corners[3][1],
            ))

        return detections

    def to_yolo_obb_line(self, det: OBBDetection) -> str:
        """转换为 YOLO OBB 格式行: class_id x1 y1 x2 y2 x3 y3 x4 y4"""
        return (
            f"{det.class_id} "
            f"{det.x1:.6f} {det.y1:.6f} "
            f"{det.x2:.6f} {det.y2:.6f} "
            f"{det.x3:.6f} {det.y3:.6f} "
            f"{det.x4:.6f} {det.y4:.6f}"
        )

    def convert_and_save(self, annotation_path: str | Path, output_path: str | Path) -> int:
        """
        读取 annotation.json，生成 YOLO OBB txt 标签文件。

        Returns
        -------
        int
            生成的检测数量。
        """
        with open(annotation_path, encoding="utf-8") as f:
            annotation = json.load(f)

        detections = self.convert_sample(annotation)
        lines = [self.to_yolo_obb_line(det) for det in detections]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return len(detections)

    def convert_dataset(
        self,
        dataset_dir: str | Path,
        split: str | None = None,
    ) -> dict[str, int]:
        """
        批量转换整个数据集。

        Parameters
        ----------
        dataset_dir : str | Path
            数据集根目录（含 manifest.json）。
        split : str | None
            如果指定，只转换该 split 中的样本（如 "train"）。

        Returns
        -------
        dict[str, int]
            {"total": N, "converted": M, "failed": F}
        """
        dataset_dir = Path(dataset_dir)
        labels_dir = dataset_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 确定要转换的样本列表
        if split:
            split_path = dataset_dir / "splits" / f"{split}.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_path}")
            with open(split_path, encoding="utf-8") as f:
                sample_ids = json.load(f)
        else:
            with open(dataset_dir / "manifest.json", encoding="utf-8") as f:
                manifest = json.load(f)
            sample_ids = [s["sample_id"] for s in manifest["samples"]]

        converted = 0
        failed = 0

        for sid in sample_ids:
            ann_path = dataset_dir / sid / "annotation.json"
            if not ann_path.exists():
                logger.warning("Annotation not found: %s", ann_path)
                failed += 1
                continue

            try:
                out_path = labels_dir / f"{sid}.txt"
                count = self.convert_and_save(ann_path, out_path)
                converted += 1
                if count == 0:
                    logger.warning("No detections for %s", sid)
            except Exception as e:
                logger.error("Failed to convert %s: %s", sid, e)
                failed += 1

        logger.info("Converted %d/%d samples (%d failed)", converted, len(sample_ids), failed)
        return {"total": len(sample_ids), "converted": converted, "failed": failed}

    @staticmethod
    def _compute_corners(
        center: np.ndarray, width: float, height: float, angle_deg: float
    ) -> list[tuple[float, float]]:
        """计算旋转矩形的四角坐标"""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        hw, hh = width / 2, height / 2
        # 局部坐标系下的四角
        local_corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

        corners = []
        for lx, ly in local_corners:
            rx = center[0] + lx * cos_a - ly * sin_a
            ry = center[1] + lx * sin_a + ly * cos_a
            corners.append((float(rx), float(ry)))

        return corners
