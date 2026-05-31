"""图像面板 — 显示原始图片和识别框叠加"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QFont, QPainter, QBrush
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

# 选中高亮样式
_HIGHLIGHT_PEN = QPen(QColor(0, 200, 255), 3, Qt.PenStyle.DashLine)
_HIGHLIGHT_BRUSH = QBrush(QColor(0, 200, 255, 40))


class ClickableRectItem(QGraphicsRectItem):
    """可点击的检测框，记录关联的元件 ref 和回调"""

    def __init__(self, ref: str, rect: QRectF, pen: QPen, callback=None, parent=None):
        super().__init__(rect, parent)
        self.ref = ref
        self._default_pen = pen
        self._callback = callback
        self.setPen(pen)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        """点击时调用回调"""
        if self._callback:
            self._callback(self.ref)
        super().mousePressEvent(event)


class ImagePanel(QWidget):
    """
    左栏图像面板。

    显示原始电路图，并叠加识别框、类别标签和置信度。
    支持点击检测框选中元件。
    """

    # 发射选中元件 ref（None 表示取消选中）
    component_selected = pyqtSignal(object)  # str | None

    # 类别颜色
    CLASS_COLORS = {
        "resistor": QColor(255, 100, 100),
        "capacitor": QColor(100, 200, 255),
        "inductor": QColor(100, 255, 100),
        "voltage_source": QColor(255, 200, 50),
        "current_source": QColor(200, 100, 255),
        "diode": QColor(255, 150, 0),
        "op_amp": QColor(150, 150, 255),
        "ground": QColor(128, 128, 128),
        "switch": QColor(0, 200, 150),
        "led": QColor(255, 80, 0),
        "npn_transistor": QColor(100, 180, 255),
        "pnp_transistor": QColor(255, 100, 180),
        "nmos": QColor(80, 220, 180),
        "pmos": QColor(220, 80, 180),
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
        self._rect_items: dict[str, ClickableRectItem] = {}  # ref -> rect item
        self._selected_ref: str | None = None

    def load_image(self, path: str):
        """加载图片"""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        self.scene.clear()
        self._overlay_items.clear()
        self._rect_items.clear()
        self._selected_ref = None

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
        self._rect_items.clear()
        self._selected_ref = None

        for det in detections:
            class_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", (0, 0, 0, 0))
            ref = det.get("ref", "")

            color = self.CLASS_COLORS.get(class_name, self.DEFAULT_COLOR)

            # 绘制可点击矩形框
            x1, y1, x2, y2 = bbox
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            pen = QPen(color, 2)
            rect_item = ClickableRectItem(ref, rect, pen, callback=self._on_rect_clicked)
            rect_item.setZValue(10)
            self.scene.addItem(rect_item)
            self._overlay_items.append(rect_item)
            if ref:
                self._rect_items[ref] = rect_item

            # 绘制标签
            label_text = f"{ref} {class_name} {conf:.0%}" if ref else f"{class_name} {conf:.0%}"
            text_item = self.scene.addText(label_text, QFont("Arial", 8))
            text_item.setDefaultTextColor(color)
            text_item.setPos(x1, y1 - 16)
            text_item.setZValue(11)
            self._overlay_items.append(text_item)

    def select_component(self, ref: str | None):
        """高亮指定检测框，传 None 取消选中"""
        # 恢复旧选中框样式
        if self._selected_ref and self._selected_ref in self._rect_items:
            old_item = self._rect_items[self._selected_ref]
            old_item.setPen(old_item._default_pen)
            old_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        self._selected_ref = ref

        # 高亮新选中框
        if ref and ref in self._rect_items:
            item = self._rect_items[ref]
            item.setPen(_HIGHLIGHT_PEN)
            item.setBrush(_HIGHLIGHT_BRUSH)

    def _on_rect_clicked(self, ref: str):
        """检测框被点击"""
        self.select_component(ref)
        self.component_selected.emit(ref)

    def wheelEvent(self, event):
        """鼠标滚轮缩放"""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.view.scale(factor, factor)
