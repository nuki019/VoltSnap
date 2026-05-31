"""电路面板 — 显示识别出的电路结构和可编辑参数"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class ComponentTableModel(QAbstractTableModel):
    """元件表格数据模型"""

    HEADERS = ["编号", "类型", "参数值", "置信度", "中心位置"]

    component_changed = pyqtSignal(str, dict)

    TYPE_DISPLAY = {
        "resistor": "电阻",
        "capacitor": "电容",
        "inductor": "电感",
        "voltage_source": "电压源",
        "current_source": "电流源",
        "diode": "二极管",
        "op_amp": "运放",
        "ground": "接地",
        "switch": "开关",
        "led": "发光二极管",
        "npn_transistor": "NPN三极管",
        "pnp_transistor": "PNP三极管",
        "nmos": "NMOS",
        "pmos": "PMOS",
    }

    def __init__(self):
        super().__init__()
        self._data: list[dict] = []

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = self._data[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return row.get("ref", "")
            elif col == 1:
                return self.TYPE_DISPLAY.get(row.get("type", ""), row.get("type", ""))
            elif col == 2:
                return row.get("value", "")
            elif col == 3:
                conf = row.get("confidence", 0)
                return f"{conf:.0%}"
            elif col == 4:
                center = row.get("center", (0, 0))
                return f"({center[0]}, {center[1]})"

        if role == Qt.ItemDataRole.BackgroundRole:
            conf = row.get("confidence", 0)
            if conf < 0.5:
                return QColor(255, 200, 200, 100)  # 低置信度标红

        if role == Qt.ItemDataRole.FontRole and col == 0:
            font = QFont()
            font.setBold(True)
            return font

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.column() in (0, 1, 2):  # 编号、类型和参数值可编辑
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.ItemDataRole.EditRole and index.isValid():
            row = index.row()
            col = index.column()
            old_ref = self._data[row].get("ref", "")
            if col == 0:
                value = str(value).strip()
                if not value:
                    return False
                self._data[row]["ref"] = value
                updates = {"ref": value}
            elif col == 1:
                value = str(value).strip()
                if not value:
                    return False
                inverse_display = {v: k for k, v in self.TYPE_DISPLAY.items()}
                value = inverse_display.get(value, value)
                self._data[row]["type"] = value
                updates = {"type": value}
            elif col == 2:
                value = str(value)
                self._data[row]["value"] = value
                updates = {"value": value}
            else:
                return False
            self.dataChanged.emit(index, index)
            self.component_changed.emit(old_ref, updates)
            return True
        return False

    def load_data(self, components: list[dict]):
        """加载元件数据"""
        self.beginResetModel()
        self._data = list(components)
        self.endResetModel()

    def get_components(self) -> list[dict]:
        """获取当前元件列表"""
        return list(self._data)

    def update_component(self, old_ref: str, updates: dict) -> bool:
        """Update a row from an external editor without re-emitting component_changed."""
        for row, comp in enumerate(self._data):
            if comp.get("ref") != old_ref:
                continue
            comp.update(updates)
            top_left = self.index(row, 0)
            bottom_right = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right)
            return True
        return False


class CircuitPanel(QWidget):
    """
    中栏电路面板。

    显示识别出的元件列表，支持编辑编号和参数值。
    """

    # 表格选中行时发射元件 ref（None 表示取消选中）
    component_selected = pyqtSignal(object)  # str | None

    component_changed = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 标题
        title = QLabel("电路结构")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # 元件表格
        group = QGroupBox("元件列表")
        group_layout = QVBoxLayout(group)

        self.table_model = ComponentTableModel()
        self.table_model.component_changed.connect(self.component_changed)
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_view.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self.table_view.setAlternatingRowColors(True)
        group_layout.addWidget(self.table_view)

        layout.addWidget(group)

        # 连接表格选择变化信号
        self.table_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )

        # 统计信息
        self.stats_label = QLabel("未加载元件")
        self.stats_label.setStyleSheet("color: gray; padding: 4px;")
        layout.addWidget(self.stats_label)

        # 操作按钮
        btn_layout = QHBoxLayout()

        btn_add = QPushButton("添加元件")
        btn_add.clicked.connect(self._on_add_component)
        btn_layout.addWidget(btn_add)

        btn_delete = QPushButton("删除选中")
        btn_delete.clicked.connect(self._on_delete_component)
        btn_layout.addWidget(btn_delete)

        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(btn_reset)

        layout.addLayout(btn_layout)

    def load_components(self, components: list[dict]):
        """加载元件列表"""
        self.table_model.load_data(components)
        # model reset 后重新连接选择信号
        self.table_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        n = len(components)
        types = {}
        for c in components:
            t = c.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        stats_parts = [f"共 {n} 个元件"]
        for t, count in types.items():
            display = ComponentTableModel.TYPE_DISPLAY.get(t, t)
            stats_parts.append(f"{display}: {count}")

        self.stats_label.setText(" | ".join(stats_parts))

    def get_components(self) -> list[dict]:
        """获取当前元件列表"""
        return self.table_model.get_components()

    def update_component(self, old_ref: str, updates: dict) -> bool:
        """从原理图编辑器同步元件改动到表格。"""
        return self.table_model.update_component(old_ref, updates)

    def _on_add_component(self):
        """添加元件"""
        row = self.table_model.rowCount()
        self.table_model.beginInsertRows(QModelIndex(), row, row)
        self.table_model._data.append({
            "ref": f"R{row + 1}",
            "type": "resistor",
            "value": "1k",
            "confidence": 1.0,
            "center": (0, 0),
        })
        self.table_model.endInsertRows()

    def _on_delete_component(self):
        """删除选中元件"""
        indexes = self.table_view.selectionModel().selectedRows()
        if not indexes:
            return

        rows = sorted([i.row() for i in indexes], reverse=True)
        for row in rows:
            self.table_model.beginRemoveRows(QModelIndex(), row, row)
            del self.table_model._data[row]
            self.table_model.endRemoveRows()

    def _on_reset(self):
        """重置为识别结果"""
        # 此方法由外部调用时传入原始数据
        pass

    def select_component(self, ref: str | None):
        """高亮指定元件行，传 None 取消选中"""
        self.table_view.clearSelection()
        if ref is None:
            return
        for row in range(self.table_model.rowCount()):
            idx = self.table_model.index(row, 0)
            if self.table_model.data(idx) == ref:
                self.table_view.selectRow(row)
                break

    def _on_selection_changed(self, selected, deselected):
        """表格选择变化时发射信号"""
        indexes = self.table_view.selectionModel().selectedRows()
        if indexes:
            row = indexes[0].row()
            ref = self.table_model.data(self.table_model.index(row, 0))
            self.component_selected.emit(ref)
        else:
            self.component_selected.emit(None)
