"""元件检测器 — YOLO OBB 模型训练与推理"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voltsnap.recognition.annotation_converter import COMPONENT_CLASSES
from voltsnap.utils import imread_unicode

logger = logging.getLogger("voltsnap.recognition.detector")


@dataclass
class Detection:
    """单个检测结果"""
    class_id: int
    class_name: str
    confidence: float
    # OBB 四角坐标（像素）
    corners: np.ndarray  # shape (4, 2)
    # 轴对齐边界框（像素）
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    # 中心点
    center: tuple[int, int]
    # 角度
    angle: float


class ComponentDetector:
    """
    元件检测器。

    使用 YOLO OBB 模型检测电路图中的元件。
    支持训练和推理两种模式。
    """

    CLASS_NAMES = {v: k for k, v in COMPONENT_CLASSES.items()}

    def __init__(self, model_path: str | None = None):
        """
        Parameters
        ----------
        model_path : str | None
            YOLO OBB 模型路径。None 则使用预训练 yolov8n-obb.pt。
        """
        self._model = None
        self._model_path = model_path

    def _ensure_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            if self._model_path:
                self._model = YOLO(self._model_path)
                logger.info("Loaded model: %s", self._model_path)
            else:
                self._model = YOLO("yolov8n-obb.pt")
                logger.info("Loaded pretrained yolov8n-obb")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[Detection]:
        """
        检测图像中的元件。

        Parameters
        ----------
        image : np.ndarray
            输入图像 (BGR, uint8)。
        conf_threshold : float
            置信度阈值。
        iou_threshold : float
            NMS IoU 阈值。

        Returns
        -------
        list[Detection]
            检测结果列表，按置信度降序排列。
        """
        self._ensure_model()
        h, w = image.shape[:2]

        results = self._model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            if result.obb is None:
                continue

            for i in range(len(result.obb)):
                obb = result.obb[i]
                cls_id = int(obb.cls[0])
                conf = float(obb.conf[0])

                # OBB 四角坐标（像素）
                if obb.xyxyxyxy is not None:
                    corners = obb.xyxyxyxy[0].cpu().numpy()
                elif obb.xywhr is not None:
                    # 从中心-宽高-角度计算四角
                    cx, cy, bw, bh, angle = obb.xywhr[0].cpu().numpy()
                    corners = self._whr_to_corners(cx, cy, bw, bh, angle)
                else:
                    continue

                # 轴对齐 bbox
                x_min = int(np.min(corners[:, 0]))
                y_min = int(np.min(corners[:, 1]))
                x_max = int(np.max(corners[:, 0]))
                y_max = int(np.max(corners[:, 1]))

                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)

                # 计算角度
                dx = corners[1][0] - corners[0][0]
                dy = corners[1][1] - corners[0][1]
                angle = float(np.degrees(np.arctan2(dy, dx)))

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self.CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
                    confidence=conf,
                    corners=corners,
                    bbox=(x_min, y_min, x_max, y_max),
                    center=(center_x, center_y),
                    angle=angle,
                ))

        # 按置信度降序排序
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_from_path(
        self,
        image_path: str | Path,
        conf_threshold: float = 0.25,
    ) -> list[Detection]:
        """从文件路径加载图像并检测"""
        image = imread_unicode(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.detect(image, conf_threshold)

    @staticmethod
    def _whr_to_corners(
        cx: float, cy: float, w: float, h: float, angle: float
    ) -> np.ndarray:
        """从中心-宽高-角度计算四角坐标"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        hw, hh = w / 2, h / 2

        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = local @ rot.T + np.array([cx, cy])
        return corners


class YOLOTrainer:
    """
    YOLO OBB 模型训练器。

    使用 ultralytics 训练自定义 OBB 模型。
    """

    def __init__(self, data_yaml: str | Path):
        self.data_yaml = str(data_yaml)

    def train(
        self,
        model_size: str = "n",
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = "0",
        project: str = "runs/obb",
        name: str = "train",
        **kwargs,
    ) -> str:
        """
        训练 YOLO OBB 模型。

        Parameters
        ----------
        model_size : str
            模型大小: n, s, m, l, x
        epochs : int
            训练轮数。
        batch_size : int
            批大小。
        img_size : int
            输入图像尺寸。
        device : str
            设备："0" 表示 GPU 0，"cpu" 表示 CPU。
        project : str
            项目保存路径。
        name : str
            实验名称。

        Returns
        -------
        str
            最佳模型路径。
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        model_name = f"yolov8{model_size}-obb.pt"
        logger.info("Starting training: %s on %s", model_name, self.data_yaml)

        model = YOLO(model_name)
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project,
            name=name,
            **kwargs,
        )

        best_path = str(Path(project) / name / "weights" / "best.pt")
        logger.info("Training complete. Best model: %s", best_path)
        return best_path


def create_data_yaml(
    dataset_dir: str | Path,
    output_path: str | Path,
    class_names: dict[int, str] | None = None,
) -> str:
    """
    生成 YOLO OBB 训练所需的 data.yaml 文件。

    Parameters
    ----------
    dataset_dir : str | Path
        数据集根目录（含 images/ 和 labels/ 子目录）。
    output_path : str | Path
        输出 yaml 路径。
    class_names : dict[int, str] | None
        类别名称映射。None 则使用默认 COMPONENT_CLASSES。

    Returns
    -------
    str
        生成的 yaml 文件路径。
    """
    import yaml

    if class_names is None:
        class_names = {v: k for k, v in COMPONENT_CLASSES.items()}

    dataset_dir = Path(dataset_dir)
    data = {
        "path": str(dataset_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
        "nc": len(class_names),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    logger.info("Created data.yaml: %s", output_path)
    return str(output_path)
