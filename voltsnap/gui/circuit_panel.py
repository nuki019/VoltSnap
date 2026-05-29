"""电路面板 — 显示识别出的电路结构和可编辑参数"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
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

    TYPE_DISPLAY = {
        "resistor": "电阻",
        "capacitor": "电容",
        "inductor": "电感",
        "voltage_source": "电压源",
        "current_source": "电流源",
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
        if index.column() in (0, 2):  # 编号和参数值可编辑
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.ItemDataRole.EditRole and index.isValid():
            row = index.row()
            col = index.column()
            if col == 0:
                self._data[row]["ref"] = value
            elif col == 2:
                self._data[row]["value"] = value
            self.dataChanged.emit(index, index)
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


class CircuitPanel(QWidget):
    """
    中栏电路面板。

    显示识别出的元件列表，支持编辑编号和参数值。
    """

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
