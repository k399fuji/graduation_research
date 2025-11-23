# python/editor/plugin_core.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow


class EditorPlugin(ABC):
    """
    すべてのプラグインのベースクラス。
    いまは「自分の QDockWidget を MainWindow に追加する」責務だけ持つ。
    """

    plugin_id: str  # 一意な ID
    display_name: str  # UI に出す名称

    @abstractmethod
    def create_dock(self, main_window: QMainWindow) -> QDockWidget:
        """
        プラグイン用の QDockWidget を作成し、必要なら main_window.addDockWidget まで行う。
        """
        raise NotImplementedError


class PluginManager:
    """
    とりあえずシンプルなプラグイン管理クラス。
    今は「プラグインクラスを登録して dock を作る」だけ。
    """

    def __init__(self, main_window: QMainWindow) -> None:
        self.main_window = main_window
        self._plugins: Dict[str, EditorPlugin] = {}
        self._docks: Dict[str, QDockWidget] = {}

    def register_plugin(self, plugin_cls: Type[EditorPlugin]) -> None:
        plugin = plugin_cls()
        pid = plugin.plugin_id
        if pid in self._plugins:
            raise ValueError(f"plugin_id '{pid}' is already registered")

        self._plugins[pid] = plugin
        dock = plugin.create_dock(self.main_window)
        self._docks[pid] = dock

    def get_plugin(self, plugin_id: str) -> EditorPlugin | None:
        return self._plugins.get(plugin_id)

    def get_dock(self, plugin_id: str) -> QDockWidget | None:
        return self._docks.get(plugin_id)
