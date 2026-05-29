"""图像面板 — 显示原始图片和识别框叠加"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ImagePanel(QWidget):
    """
    左栏图像面板。

    显示原始电路图，并叠加识别框、类别标签和置信度。
    """

    # 类别颜色
    CLASS_COLORS = {
        "resistor": QColor(255, 100, 100),
        "capacitor": QColor(100, 200, 255),
        "inductor": QColor(100, 255, 100),
        "voltage_source": QColor(255, 200, 50),
        "current_source": QColor(200, 100, 255),
    }
    DEFAULT_COLOR = QColor(255, 255, 255)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("电路图")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.title_label)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        layout.addWidget(self.view)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: list = []

    def load_image(self, path: str):
        """加载图片"""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        self.scene.clear()
        self._overlay_items.clear()

        self._pixmap_item = self.scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self.scene.setSceneRect(pixmap.rect())

        self.view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.title_label.setText(f"电路图 — {path.split('/')[-1]}")

    def overlay_detections(self, detections: list[dict]):
        """叠加识别框"""
        # 清除旧的叠加层
        for item in self._overlay_items:
            self.scene.removeItem(item)
        self._overlay_items.clear()

        for det in detections:
            class_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", (0, 0, 0, 0))

            color = self.CLASS_COLORS.get(class_name, self.DEFAULT_COLOR)

            # 绘制矩形框
            x1, y1, x2, y2 = bbox
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(color, 2)
            rect_item = self.scene.addRect(rect, pen)
            rect_item.setZValue(10)
            self._overlay_items.append(rect_item)

            # 绘制标签
            label_text = f"{class_name} {conf:.0%}"
            text_item = self.scene.addText(label_text, QFont("Arial", 8))
            text_item.setDefaultTextColor(color)
            text_item.setPos(x1, y1 - 16)
            text_item.setZValue(11)
            self._overlay_items.append(text_item)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.view.scale(factor, factor)
