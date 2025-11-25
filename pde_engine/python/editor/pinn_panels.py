# editor/pinn_panels.py
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTabWidget,
    QDockWidget,
    QMainWindow,
)

from editor.plugin_core import EditorPlugin, register_builtin_plugin

# ==============================
# タブ付き PINN プラグイン
# ==============================

class PINNTabbedPlugin(EditorPlugin):
    plugin_id = "pinn_tabbed"
    display_name = "PINN"

    def create_dock(self, main_window: QMainWindow) -> QDockWidget:
        from editor.heat_panel import HeatPINNPanel
        from editor.wave_panel import WavePINNPanel

        tab_widget = QTabWidget(main_window)

        heat_page = HeatPINNPanel(tab_widget)
        wave_page = WavePINNPanel(tab_widget)

        tab_widget.addTab(heat_page, "Heat")
        tab_widget.addTab(wave_page, "Wave")

        if hasattr(main_window, "on_pinn_run_finished"):
            heat_page.runFinished.connect(main_window.on_pinn_run_finished)
            wave_page.runFinished.connect(main_window.on_pinn_run_finished)

        dock = QDockWidget(self.display_name, main_window)
        dock.setWidget(tab_widget)
        dock.setObjectName(self.plugin_id)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        main_window.addDockWidget(Qt.LeftDockWidgetArea, dock)

        if hasattr(main_window, "view_menu"):
            main_window.view_menu.addAction(dock.toggleViewAction())

        main_window.pinn_dock = dock
        dock.hide()
        return dock

# ★ ここを追加：プラグイン Factory に登録
register_builtin_plugin(PINNTabbedPlugin)
