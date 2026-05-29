"""VoltSnap 桌面应用入口"""
from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from voltsnap.gui.main_window import MainWindow


def main():
    """启动 VoltSnap 桌面应用"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("VoltSnap")
    app.setOrganizationName("VoltSnap")

    # 全局样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QSplitter::handle {
            background-color: #555;
            width: 3px;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 16px;
            font-weight: bold;
            color: #ddd;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QLabel {
            color: #ccc;
        }
        QTableView {
            background-color: #1e1e1e;
            alternate-background-color: #252525;
            color: #ddd;
            gridline-color: #444;
            selection-background-color: #3a6ea5;
        }
        QHeaderView::section {
            background-color: #333;
            color: #ddd;
            padding: 4px;
            border: 1px solid #555;
        }
        QPlainTextEdit {
            background-color: #1e1e1e;
            color: #d4d4d4;
            border: 1px solid #555;
            font-family: Consolas, monospace;
        }
        QPushButton {
            background-color: #3c3c3c;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 6px 16px;
            min-height: 24px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #555;
        }
        QToolBar {
            background-color: #333;
            border-bottom: 1px solid #555;
            spacing: 6px;
            padding: 4px;
        }
        QToolBar QToolButton {
            color: #ddd;
            padding: 4px 8px;
        }
        QStatusBar {
            background-color: #2b2b2b;
            color: #aaa;
        }
        QDockWidget {
            color: #ddd;
        }
        QDockWidget::title {
            background-color: #333;
            padding: 4px;
        }
        QTabWidget::pane {
            border: 1px solid #555;
        }
        QTabBar::tab {
            background-color: #333;
            color: #ddd;
            padding: 6px 16px;
            border: 1px solid #555;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background-color: #2b2b2b;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 4px;
            text-align: center;
            color: #ddd;
        }
        QProgressBar::chunk {
            background-color: #3a6ea5;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
