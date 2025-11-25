# python/editor/plugin_core.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow


# =========================================================
# EditorPlugin Base
# =========================================================

class EditorPlugin(ABC):
    """
    すべてのプラグインの基底クラス。
    役割：
        - 自分の QDockWidget を作る
        - MainWindow に Dock を追加する責務は create_dock 内で行う
    """

    plugin_id: str      # 一意な ID
    display_name: str   # UI 表示名

    @abstractmethod
    def create_dock(self, main_window: QMainWindow) -> QDockWidget:
        raise NotImplementedError


# =========================================================
# PluginManager
# =========================================================

class PluginManager:
    """
    プラグイン登録管理。
    main_window.register_builtin_plugins() はやめ、
    この PluginManager 自体に「ビルトイン登録」を持たせる。
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

    def register_many(self, plugin_list: List[Type[EditorPlugin]]) -> None:
        """複数一括登録用"""
        for cls in plugin_list:
            self.register_plugin(cls)

    def get_plugin(self, plugin_id: str) -> EditorPlugin | None:
        return self._plugins.get(plugin_id)

    def get_dock(self, plugin_id: str) -> QDockWidget | None:
        return self._docks.get(plugin_id)


# =========================================================
# Built-in Plugin Factory
# =========================================================

_BUILTIN_PLUGINS: List[Type[EditorPlugin]] = []


def register_builtin_plugin(plugin_cls: Type[EditorPlugin]) -> None:
    """
    他ファイルから提供されるプラグインをグローバル登録する。
    main_qt.py から import せずに済むようにする。
    """
    _BUILTIN_PLUGINS.append(plugin_cls)


def get_builtin_plugins() -> List[Type[EditorPlugin]]:
    """
    main_qt.py が唯一呼べばよいインターフェース。
    """
    return list(_BUILTIN_PLUGINS)
