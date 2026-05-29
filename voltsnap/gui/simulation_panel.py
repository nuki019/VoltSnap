"""仿真面板 — 网表编辑 + 仿真结果 + 波形展示"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class SimulationPanel(QWidget):
    """
    右栏仿真面板。

    包含：
    - 网表编辑器
    - 节点电压表
    - 波形图（pyqtgraph）
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 标题
        title = QLabel("仿真")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        # Tab 切换
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: 网表编辑
        netlist_tab = QWidget()
        netlist_layout = QVBoxLayout(netlist_tab)
        netlist_layout.setContentsMargins(0, 0, 0, 0)

        self.netlist_editor = QPlainTextEdit()
        self.netlist_editor.setFont(QFont("Consolas", 10))
        self.netlist_editor.setPlaceholderText("SPICE 网表将在此显示...")
        netlist_layout.addWidget(self.netlist_editor)

        tabs.addTab(netlist_tab, "网表")

        # Tab 2: 仿真结果
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        result_layout.setContentsMargins(0, 0, 0, 0)

        self.voltage_table = QTableWidget()
        self.voltage_table.setColumnCount(2)
        self.voltage_table.setHorizontalHeaderLabels(["节点", "电压 (V)"])
        self.voltage_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.voltage_table.setAlternatingRowColors(True)
        result_layout.addWidget(QLabel("节点电压:"))
        result_layout.addWidget(self.voltage_table)

        self.current_table = QTableWidget()
        self.current_table.setColumnCount(2)
        self.current_table.setHorizontalHeaderLabels(["支路", "电流 (A)"])
        self.current_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.current_table.setAlternatingRowColors(True)
        result_layout.addWidget(QLabel("支路电流:"))
        result_layout.addWidget(self.current_table)

        tabs.addTab(result_tab, "结果")

        # Tab 3: 波形
        waveform_tab = QWidget()
        waveform_layout = QVBoxLayout(waveform_tab)
        waveform_layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            pg.setConfigOptions(antialias=True)
            self.plot_widget = pg.PlotWidget(title="节点电压")
            self.plot_widget.setLabel("left", "电压", units="V")
            self.plot_widget.setLabel("bottom", "节点")
            self.plot_widget.addLegend()
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            waveform_layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            waveform_layout.addWidget(QLabel("pyqtgraph 未安装，波形不可用"))

        tabs.addTab(waveform_tab, "波形")

        # 状态
        self.status_label = QLabel("等待仿真")
        self.status_label.setStyleSheet("color: gray; padding: 2px;")
        layout.addWidget(self.status_label)

    # ── 网表操作 ──────────────────────────────────────────────────────

    def set_netlist(self, netlist: str):
        """设置网表内容"""
        self.netlist_editor.setPlainText(netlist)

    def get_netlist(self) -> str:
        """获取网表内容"""
        return self.netlist_editor.toPlainText()

    # ── 结果展示 ──────────────────────────────────────────────────────

    def show_results(self, result):
        """
        展示仿真结果。

        Parameters
        ----------
        result : SimulationResult
            ngspice 仿真结果。
        """
        # 节点电压表
        voltages = result.node_voltages
        self.voltage_table.setRowCount(len(voltages))
        for i, (node, voltage) in enumerate(sorted(voltages.items())):
            self.voltage_table.setItem(i, 0, QTableWidgetItem(node))
            item = QTableWidgetItem(f"{voltage:.6f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.voltage_table.setItem(i, 1, item)

        # 支路电流表
        currents = result.branch_currents
        self.current_table.setRowCount(len(currents))
        for i, (branch, current) in enumerate(sorted(currents.items())):
            self.current_table.setItem(i, 0, QTableWidgetItem(branch))
            item = QTableWidgetItem(f"{current:.6e}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.current_table.setItem(i, 1, item)

        # 波形图
        if self.plot_widget and voltages:
            self.plot_widget.clear()
            nodes = sorted(voltages.keys())
            values = [voltages[n] for n in nodes]

            bar_plot = pg.BarGraphItem(
                x=range(len(nodes)),
                height=values,
                width=0.6,
                brush=pg.mkBrush(100, 150, 255, 150),
                pen=pg.mkPen(50, 100, 200),
            )
            self.plot_widget.addItem(bar_plot)
            self.plot_widget.getAxis("bottom").setTicks(
                [[(i, n) for i, n in enumerate(nodes)]]
            )

        # 状态
        self.status_label.setText(
            f"仿真成功 | {len(voltages)} 个节点 | {len(currents)} 条支路"
        )
        self.status_label.setStyleSheet("color: green; padding: 2px;")

    def clear_results(self):
        """清除结果"""
        self.voltage_table.setRowCount(0)
        self.current_table.setRowCount(0)
        if self.plot_widget:
            self.plot_widget.clear()
        self.status_label.setText("等待仿真")
        self.status_label.setStyleSheet("color: gray; padding: 2px;")
