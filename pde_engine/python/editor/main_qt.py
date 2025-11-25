from __future__ import annotations

import sys
import os

from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QDockWidget,
)

# プロジェクトの python/ を import パスに追加
CURRENT_DIR = os.path.dirname(__file__)
PYTHON_DIR = os.path.dirname(CURRENT_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.append(PYTHON_DIR)

from editor.plugin_core import PluginManager, get_builtin_plugins
from editor.viewport_1d import ViewportWidget
from editor.heat2d_tab import Heat2DTab
from editor.home_page import HomePage

import editor.pinn_panels #プラグイン登録用：参照されてないけど消したらダメ
import editor.logs_panel 


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    app.setPalette(palette)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDE Engine & PINN Editor (Qt)")
        self.pinn_dock: QDockWidget | None = None

        # タブ全体
        self.tabs = QTabWidget(self)

        # Home
        self.home_page = HomePage(self)
        self.tabs.addTab(self.home_page, "Home")

        # PINN (central viewport)
        self.viewport = ViewportWidget(self)
        self.tabs.addTab(self.viewport, "PINN")

        # 2D Heat
        self.heat2d_tab = Heat2DTab(self)
        self.tabs.addTab(self.heat2d_tab, "2D Heat")

        self.setCentralWidget(self.tabs)

        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.home_page.openPinnRequested.connect(self.open_pinn_workspace)
        self.home_page.openHeat2DRequested.connect(
            lambda: self.tabs.setCurrentWidget(self.heat2d_tab)
        )

        menu_bar = self.menuBar()
        self.view_menu = menu_bar.addMenu("&View")

        # プラグイン
        self.plugin_manager = PluginManager(self)
        self.plugin_manager.register_many(get_builtin_plugins())
        self.pinn_dock = self.plugin_manager.get_dock("pinn_tabbed")

        self.resize(1200, 800)

    @Slot(str)
    def on_pinn_run_finished(self, run_id: str):
        self.viewport.show_result(run_id)

    @Slot(str)
    def on_log_run_selected(self, run_id: str):
        self.viewport.show_result(run_id)

    @Slot()
    def open_pinn_workspace(self):
        self.tabs.setCurrentWidget(self.viewport)

        if self.pinn_dock is not None:
            self.pinn_dock.show()
            self.pinn_dock.raise_()

    @Slot(int)
    def on_tab_changed(self, index: int):
        if self.pinn_dock is None:
            return
        current_widget = self.tabs.widget(index)
        if current_widget is self.home_page:
            self.pinn_dock.hide()
        elif current_widget is self.viewport:
            self.pinn_dock.show()
            self.pinn_dock.raise_()


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
