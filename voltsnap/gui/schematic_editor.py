"""原理图编辑器 — 以标准电气符号渲染识别出的元件，支持仿真结果叠加"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# ── 样式常量 ──────────────────────────────────────────────────────────────

_GRID_PEN = QPen(QColor(60, 60, 60), 0.5)
_WIRE_PEN = QPen(QColor(200, 200, 200), 1.5)
_SYMBOL_PEN = QPen(QColor(220, 220, 220), 2)
_FILL_BRUSH = QBrush(QColor(40, 40, 40))

_HIGHLIGHT_PEN = QPen(QColor(0, 200, 255), 3, Qt.PenStyle.DashLine)
_HIGHLIGHT_BRUSH = QBrush(QColor(0, 200, 255, 40))

_LABEL_FONT = QFont("Consolas", 9)
_VALUE_FONT = QFont("Consolas", 8)

_OVERLAY_BG = QBrush(QColor(0, 0, 0, 180))
_OVERLAY_PEN = QPen(QColor(0, 200, 255), 1)

# 每个元件占用的逻辑尺寸
COMP_W = 80.0  # 宽度（元件体 + 引脚线）
COMP_H = 40.0  # 高度


# ── 元件图形项 ────────────────────────────────────────────────────────────

class ComponentItem(QGraphicsItem):
    """可点击的元件符号，包含引脚延长线、符号体和 ref/value 标签。

    坐标系：以元件体中心为原点 (0, 0)。
    左侧引脚线从 (-COMP_W/2, 0) 到体左端；
    右侧引脚线从体右端到 (COMP_W/2, 0)。
    """

    def __init__(self, comp: dict, on_click=None):
        super().__init__()
        self.comp = comp
        self.ref = comp.get("ref", "")
        self.comp_type = comp.get("type", "unknown")
        self.value = comp.get("value", "")
        self._on_click = on_click
        self._selected = False
        self._sim_texts: list[QGraphicsSimpleTextItem] = []

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # 子图形项列表，用于 hitTest 和颜色管理
        self._body_items: list[QGraphicsItem] = []
        self._label_items: list[QGraphicsItem] = []
        self._build_symbol()

    # ── 符号构建 ──────────────────────────────────────────────────────

    def _build_symbol(self):
        """根据类型创建符号子项"""
        hw = COMP_W / 2
        hh = COMP_H / 2

        # 引脚延长线（左右各 15px）
        pin_len = 15.0
        left_line = QGraphicsLineItem(-hw, 0, -hw + pin_len, 0, self)
        left_line.setPen(_WIRE_PEN)
        right_line = QGraphicsLineItem(hw - pin_len, 0, hw, 0, self)
        right_line.setPen(_WIRE_PEN)

        # 小圆圈引脚
        pin_r = 3.0
        lp = QGraphicsEllipseItem(-hw - pin_r, -pin_r, pin_r * 2, pin_r * 2, self)
        lp.setPen(_WIRE_PEN)
        lp.setBrush(QBrush(QColor(200, 200, 200)))
        rp = QGraphicsEllipseItem(hw - pin_r, -pin_r, pin_r * 2, pin_r * 2, self)
        rp.setPen(_WIRE_PEN)
        rp.setBrush(QBrush(QColor(200, 200, 200)))

        builder = _SYMBOL_BUILDERS.get(self.comp_type, _build_unknown)
        body_items = builder(self, -hw + pin_len, hw - pin_len, -hh, hh)
        for item in body_items:
            item.setPen(_SYMBOL_PEN)
            if isinstance(item, (QGraphicsRectItem, QGraphicsEllipseItem,
                                 QGraphicsPolygonItem, QGraphicsPathItem)):
                item.setBrush(_FILL_BRUSH)
            self._body_items.append(item)

        # ref 标签（上方）
        ref_label = QGraphicsSimpleTextItem(self.ref, self)
        ref_label.setFont(_LABEL_FONT)
        ref_label.setBrush(QBrush(QColor(255, 220, 100)))
        ref_label.setPos(-ref_label.boundingRect().width() / 2, -hh - 18)
        self._label_items.append(ref_label)

        # value 标签（下方）
        if self.value:
            val_label = QGraphicsSimpleTextItem(self.value, self)
            val_label.setFont(_VALUE_FONT)
            val_label.setBrush(QBrush(QColor(180, 180, 180)))
            val_label.setPos(-val_label.boundingRect().width() / 2, hh + 4)
            self._label_items.append(val_label)

    # ── 选中状态 ──────────────────────────────────────────────────────

    def set_selected(self, selected: bool):
        """设置选中高亮状态"""
        if self._selected == selected:
            return
        self._selected = selected
        pen = _HIGHLIGHT_PEN if selected else _SYMBOL_PEN
        brush = _HIGHLIGHT_BRUSH if selected else _FILL_BRUSH
        for item in self._body_items:
            item.setPen(pen)
            if isinstance(item, (QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPolygonItem, QGraphicsPathItem)):
                item.setBrush(brush)
        self.update()

    # ── 仿真结果标签 ──────────────────────────────────────────────────

    def set_simulation_labels(self, texts: list[str]):
        """在元件附近叠加仿真结果标签"""
        clear_simulation_labels(self._sim_texts)
        if not texts:
            return
        hh = COMP_H / 2
        y = hh + 18
        for i, txt in enumerate(texts):
            bg = QGraphicsRectItem(self)
            label = QGraphicsSimpleTextItem(txt, self)
            label.setFont(_VALUE_FONT)
            label.setBrush(QBrush(QColor(0, 230, 180)))
            lr = label.boundingRect()
            label.setPos(-lr.width() / 2, y + i * 16)
            bg.setRect(label.x() - 2, label.y() - 1, lr.width() + 4, lr.height() + 2)
            bg.setBrush(_OVERLAY_BG)
            bg.setPen(_OVERLAY_PEN)
            self._sim_texts.append(bg)
            self._sim_texts.append(label)

    # ── QGraphicsItem 接口 ────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        hw = COMP_W / 2
        hh = COMP_H / 2
        return QRectF(-hw - 5, -hh - 22, COMP_W + 10, COMP_H + 50)

    def paint(self, painter, option, widget=None):
        pass  # 所有绘制由子项完成

    def mousePressEvent(self, event):
        if self._on_click:
            self._on_click(self.ref)
        super().mousePressEvent(event)


def clear_simulation_labels(items: list):
    """从场景中移除仿真标签子项"""
    for item in items:
        scene = item.scene()
        if scene:
            scene.removeItem(item)
    items.clear()


# ── 各类型符号构建器 ──────────────────────────────────────────────────────

def _build_resistor(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """电阻 — 锯齿形折线"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    seg_w = (x2 - x1) / 7
    amp = (y2 - y1) / 2 * 0.8

    points = [QPointF(x1, cy)]
    for i in range(7):
        px = x1 + seg_w * (i + 0.5)
        py = cy + (amp if i % 2 == 0 else -amp)
        points.append(QPointF(px, py))
    points.append(QPointF(x2, cy))

    path = QPainterPath()
    path.moveTo(points[0])
    for pt in points[1:]:
        path.lineTo(pt)
    item = QGraphicsPathItem(path, comp)
    return [item]


def _build_capacitor(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """电容 — 两平行线"""
    cx = (x1 + x2) / 2
    gap = 6.0
    items = []
    items.append(QGraphicsLineItem(x1, (y1 + y2) / 2, cx - gap, (y1 + y2) / 2, comp))
    items.append(QGraphicsLineItem(cx + gap, (y1 + y2) / 2, x2, (y1 + y2) / 2, comp))
    items.append(QGraphicsLineItem(cx - gap, y1 + 4, cx - gap, y2 - 4, comp))
    items.append(QGraphicsLineItem(cx + gap, y1 + 4, cx + gap, y2 - 4, comp))
    return items


def _build_inductor(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """电感 — 弧线（用半圆近似）"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    n_bumps = 4
    seg_w = (x2 - x1) / n_bumps
    r = seg_w / 2

    path = QPainterPath()
    path.moveTo(x1, cy)
    for i in range(n_bumps):
        bx = x1 + seg_w * i
        arc = QRectF(bx, cy - r, seg_w, r * 2)
        path.arcTo(arc, 180, -180)
    item = QGraphicsPathItem(path, comp)
    return [item]


def _build_voltage_source(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """电压源 — 圆 + +/−"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r = min(x2 - x1, y2 - y1) / 2 - 2

    items = []
    circle = QGraphicsEllipseItem(cx - r, cy - r, r * 2, r * 2, comp)
    items.append(circle)

    # + 号（左侧）
    plus_len = 5
    items.append(QGraphicsLineItem(cx - r + 4, cy, cx - r + 4 + plus_len, cy, comp))
    items.append(QGraphicsLineItem(cx - r + 4 + plus_len / 2, cy - plus_len / 2,
                                    cx - r + 4 + plus_len / 2, cy + plus_len / 2, comp))

    # − 号（右侧）
    items.append(QGraphicsLineItem(cx + r - 4 - plus_len, cy, cx + r - 4, cy, comp))
    return items


def _build_current_source(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """电流源 — 圆 + 箭头"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r = min(x2 - x1, y2 - y1) / 2 - 2

    items = []
    circle = QGraphicsEllipseItem(cx - r, cy - r, r * 2, r * 2, comp)
    items.append(circle)

    # 箭头（从左到右）
    arrow_len = r * 1.2
    items.append(QGraphicsLineItem(cx - arrow_len / 2, cy, cx + arrow_len / 2, cy, comp))

    # 箭头头部
    head_len = 6
    head = QPolygonF([
        QPointF(cx + arrow_len / 2, cy),
        QPointF(cx + arrow_len / 2 - head_len, cy - head_len / 2),
        QPointF(cx + arrow_len / 2 - head_len, cy + head_len / 2),
    ])
    arrow_head = QGraphicsPolygonItem(head, comp)
    items.append(arrow_head)
    return items


def _build_ground(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """接地 — 三横线递减"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    items = []
    # 垂直线
    items.append(QGraphicsLineItem(cx, y1 + 4, cx, cy, comp))
    # 三条横线
    for i, w in enumerate([18, 12, 6]):
        yy = cy + i * 5
        items.append(QGraphicsLineItem(cx - w / 2, yy, cx + w / 2, yy, comp))
    return items


def _build_diode(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """二极管 — 三角形 + 竖线"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    tri_half = min(x2 - x1, y2 - y1) / 2 - 4

    items = []
    # 连接线
    items.append(QGraphicsLineItem(x1, cy, cx - tri_half, cy, comp))
    items.append(QGraphicsLineItem(cx + tri_half, cy, x2, cy, comp))

    # 三角形
    tri = QPolygonF([
        QPointF(cx - tri_half, cy - tri_half),
        QPointF(cx - tri_half, cy + tri_half),
        QPointF(cx + tri_half, cy),
    ])
    items.append(QGraphicsPolygonItem(tri, comp))

    # 竖线（阴极）
    items.append(QGraphicsLineItem(cx + tri_half, cy - tri_half, cx + tri_half, cy + tri_half, comp))
    return items


def _build_led(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """LED — 同二极管 + 两个小箭头表示发光"""
    items = _build_diode(comp, x1, x2, y1, y2)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # 发光箭头
    for dy in (-10, 14):
        arr = QPolygonF([
            QPointF(cx + 12, cy + dy),
            QPointF(cx + 20, cy + dy - 4),
            QPointF(cx + 16, cy + dy - 2),
        ])
        items.append(QGraphicsPolygonItem(arr, comp))
    return items


def _build_op_amp(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """运放 — 三角形"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    tri_half = min(x2 - x1, y2 - y1) / 2 - 2

    items = []
    # 连接线
    items.append(QGraphicsLineItem(x1, cy, cx - tri_half, cy, comp))
    items.append(QGraphicsLineItem(cx + tri_half, cy, x2, cy, comp))

    # 三角形
    tri = QPolygonF([
        QPointF(cx - tri_half, y1 + 2),
        QPointF(cx - tri_half, y2 - 2),
        QPointF(cx + tri_half, cy),
    ])
    items.append(QGraphicsPolygonItem(tri, comp))
    return items


def _build_switch(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """开关 — 断开的线段 + 触点"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    items = []
    items.append(QGraphicsLineItem(x1, cy, cx - 8, cy, comp))
    items.append(QGraphicsLineItem(cx + 8, cy, x2, cy, comp))

    # 触点（圆）
    r = 3
    items.append(QGraphicsEllipseItem(cx - 8 - r, cy - r, r * 2, r * 2, comp))
    items.append(QGraphicsEllipseItem(cx + 8 - r, cy - r, r * 2, r * 2, comp))

    # 断开臂
    items.append(QGraphicsLineItem(cx - 8, cy, cx + 6, cy - 12, comp))
    return items


def _build_transistor(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """三极管 — 竖线 + 基极箭头"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    items = []
    # 基极线
    items.append(QGraphicsLineItem(x1, cy, cx - 6, cy, comp))
    # 竖线（基极-发射极界面）
    items.append(QGraphicsLineItem(cx - 6, y1 + 6, cx - 6, y2 - 6, comp))
    # 集电极
    items.append(QGraphicsLineItem(cx - 6, cy - 8, x2, y1 + 4, comp))
    # 发射极 + 箭头
    items.append(QGraphicsLineItem(cx - 6, cy + 8, x2, y2 - 4, comp))

    # 箭头
    is_pnp = "pnp" in comp.comp_type
    head_len = 5
    if is_pnp:
        arr = QPolygonF([
            QPointF(cx - 6, cy + 8),
            QPointF(cx - 6 + head_len, cy + 8 + head_len / 2),
            QPointF(cx - 6 + head_len, cy + 8 - head_len / 2),
        ])
    else:
        arr = QPolygonF([
            QPointF(x2, y2 - 4),
            QPointF(x2 - head_len, y2 - 4 + head_len / 2),
            QPointF(x2 - head_len, y2 - 4 - head_len / 2),
        ])
    items.append(QGraphicsPolygonItem(arr, comp))
    return items


def _build_mos(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """MOSFET — 简化符号"""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    items = []
    # 栅极
    items.append(QGraphicsLineItem(x1, cy, cx - 8, cy, comp))
    # 栅极竖线
    items.append(QGraphicsLineItem(cx - 8, y1 + 6, cx - 8, y2 - 6, comp))
    # 沟道竖线
    items.append(QGraphicsLineItem(cx - 2, y1 + 8, cx - 2, y2 - 8, comp))
    # 漏极
    items.append(QGraphicsLineItem(cx - 2, y1 + 8, x2, y1 + 4, comp))
    # 源极
    items.append(QGraphicsLineItem(cx - 2, y2 - 8, x2, y2 - 4, comp))
    # 衬底引线
    items.append(QGraphicsLineItem(cx + 4, cy, x2, cy, comp))
    return items


def _build_unknown(comp: ComponentItem, x1, x2, y1, y2) -> list[QGraphicsItem]:
    """未知元件 — 矩形"""
    rect = QGraphicsRectItem(x1 + 4, y1 + 4, x2 - x1 - 8, y2 - y1 - 8, comp)
    return [rect]


# 类型 → 构建器映射
_SYMBOL_BUILDERS = {
    "resistor": _build_resistor,
    "capacitor": _build_capacitor,
    "inductor": _build_inductor,
    "voltage_source": _build_voltage_source,
    "current_source": _build_current_source,
    "ground": _build_ground,
    "diode": _build_diode,
    "led": _build_led,
    "op_amp": _build_op_amp,
    "switch": _build_switch,
    "npn_transistor": _build_transistor,
    "pnp_transistor": _build_transistor,
    "nmos": _build_mos,
    "pmos": _build_mos,
}


# ── 原理图编辑器主组件 ────────────────────────────────────────────────────

class SchematicEditor(QWidget):
    """
    原理图编辑器。

    接收识别结果（list[dict]），以标准电气符号渲染元件。
    支持点击选择、双向选择联动和仿真结果叠加。
    """

    component_selected = pyqtSignal(object)  # str | None

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("原理图")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.title_label)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.view.setStyleSheet("background-color: #2b2b2b;")
        layout.addWidget(self.view)

        self._comp_items: dict[str, ComponentItem] = {}
        self._selected_ref: str | None = None
        self._components: list[dict] = []
        self._overlay_items: list[QGraphicsItem] = []

    # ── 数据加载 ──────────────────────────────────────────────────────

    def load_components(self, components: list[dict]):
        """加载元件并重新渲染原理图"""
        self.scene.clear()
        self._comp_items.clear()
        self._overlay_items.clear()
        self._selected_ref = None
        self._components = [dict(c) for c in components]

        if not components:
            self.title_label.setText("原理图 — 无元件")
            return

        # 基于 center 坐标布局
        centers = [c.get("center", (0, 0)) for c in components]
        cx_vals = [p[0] for p in centers]
        cy_vals = [p[1] for p in centers]

        # 计算缩放因子，使元件均匀分布
        if len(components) == 1:
            scale = 1.0
        else:
            range_x = max(cx_vals) - min(cx_vals) or 1
            range_y = max(cy_vals) - min(cy_vals) or 1
            target_w = max(400, len(components) * 120)
            target_h = max(300, len(components) * 80)
            scale = min(target_w / range_x, target_h / range_y, 2.0)
            scale = max(scale, 0.3)

        # 计算坐标范围用于居中偏移
        avg_x = sum(cx_vals) / len(cx_vals)
        avg_y = sum(cy_vals) / len(cy_vals)

        for comp in components:
            center = comp.get("center", (0, 0))
            # 将原始坐标映射到场景坐标，居中放置
            sx = (center[0] - avg_x) * scale
            sy = (center[1] - avg_y) * scale

            item = ComponentItem(comp, on_click=self._on_component_clicked)
            item.setPos(sx, sy)
            self.scene.addItem(item)
            ref = comp.get("ref", "")
            if ref:
                self._comp_items[ref] = item

        # 适配视图
        self.view.fitInView(self.scene.sceneRect().adjusted(-40, -40, 40, 40),
                            Qt.AspectRatioMode.KeepAspectRatio)
        self.title_label.setText(f"原理图 — {len(components)} 个元件")

    def reload_components(self, components: list[dict]):
        """重新加载元件列表（别名）"""
        self.load_components(components)

    def get_components(self) -> list[dict]:
        """获取当前元件列表"""
        return list(self._components)

    # ── 单元件刷新 ────────────────────────────────────────────────────

    def refresh_component(self, ref: str, updates: dict):
        """刷新指定元件的属性（ref/value 等），重新生成标签"""
        if ref not in self._comp_items:
            return

        # 更新内部数据
        for comp in self._components:
            if comp.get("ref") == ref:
                comp.update(updates)
                break

        item = self._comp_items.pop(ref)
        new_ref = updates.get("ref", ref)
        new_comp_data = dict(item.comp)
        new_comp_data.update(updates)

        # 移除旧项，创建新项
        pos = item.pos()
        self.scene.removeItem(item)
        new_item = ComponentItem(new_comp_data, on_click=self._on_component_clicked)
        new_item.setPos(pos)
        self.scene.addItem(new_item)
        self._comp_items[new_ref] = new_item

        # 恢复选中状态
        if self._selected_ref == ref:
            self._selected_ref = new_ref
            new_item.set_selected(True)

    # ── 选择联动 ──────────────────────────────────────────────────────

    def select_component(self, ref: str | None):
        """高亮指定元件，传 None 取消选中"""
        # 取消旧选中
        if self._selected_ref and self._selected_ref in self._comp_items:
            self._comp_items[self._selected_ref].set_selected(False)

        self._selected_ref = ref

        if ref and ref in self._comp_items:
            self._comp_items[ref].set_selected(True)

    def _on_component_clicked(self, ref: str):
        """元件被点击"""
        self.select_component(ref)
        self.component_selected.emit(ref)

    # ── 仿真结果叠加 ──────────────────────────────────────────────────

    def show_simulation_result(self, result):
        """叠加仿真结果到元件上

        Parameters
        ----------
        result : SimulationResult
            包含 node_voltages 和 branch_currents 的仿真结果。
        """
        node_v = result.node_voltages if hasattr(result, "node_voltages") else {}
        branch_i = result.branch_currents if hasattr(result, "branch_currents") else {}
        self.clear_simulation_overlay()

        for ref, item in self._comp_items.items():
            texts = []

            # 匹配支路电流: key 格式如 "v1#branch" 或 "i(v1)"
            ref_lower = ref.lower()
            for key, current in branch_i.items():
                key_lower = key.lower()
                if ref_lower in key_lower or key_lower in ref_lower:
                    texts.append(f"i({ref})={current:.4g}A")
                    break

            # 匹配节点电压: key 格式如 "v(n1)"
            for key, voltage in node_v.items():
                key_lower = key.lower()
                if ref_lower in key_lower or key_lower in ref_lower:
                    texts.append(f"V={voltage:.4g}V")
                    break

            item.set_simulation_labels(texts)

        self._add_simulation_summary(node_v, branch_i)

    def clear_simulation_overlay(self):
        """清除所有仿真叠加标签"""
        for item in self._comp_items.values():
            item.set_simulation_labels([])
        for overlay_item in self._overlay_items:
            scene = overlay_item.scene()
            if scene:
                scene.removeItem(overlay_item)
        self._overlay_items.clear()

    # ── 鼠标滚轮缩放 ─────────────────────────────────────────────────

    def _add_simulation_summary(self, node_voltages: dict, branch_currents: dict):
        lines = []
        for node, voltage in sorted(node_voltages.items()):
            lines.append(f"{node}: {voltage:.4g} V")
        for branch, current in sorted(branch_currents.items()):
            lines.append(f"{branch}: {current:.4g} A")
        if not lines:
            return

        label = QGraphicsSimpleTextItem("Simulation\n" + "\n".join(lines[:10]))
        label.setFont(_VALUE_FONT)
        label.setBrush(QBrush(QColor(0, 230, 180)))

        scene_rect = self.scene.itemsBoundingRect()
        x = scene_rect.left() if scene_rect.isValid() else 0
        y = scene_rect.top() if scene_rect.isValid() else 0
        label.setPos(x, y - label.boundingRect().height() - 12)

        rect = label.boundingRect()
        bg = QGraphicsRectItem(
            label.x() - 6,
            label.y() - 5,
            rect.width() + 12,
            rect.height() + 10,
        )
        bg.setBrush(_OVERLAY_BG)
        bg.setPen(_OVERLAY_PEN)
        bg.setZValue(90)
        label.setZValue(91)

        self.scene.addItem(bg)
        self.scene.addItem(label)
        self._overlay_items.extend([bg, label])

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.view.scale(factor, factor)
