# editor/heat2d_logs_panel.py
from __future__ import annotations

import json
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QListWidget, QPushButton, QTextEdit, QSlider,
    QDockWidget,
)

# Heat2D シミュレーション用ログユーティリティ
from backend import sim_log_utils as log_utils

from editor.heat2d_tab import Heat2DCanvas
from editor.plugin_core import EditorPlugin, register_builtin_plugin


class Heat2DLogsPage(QWidget):
    runSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_run_id: Optional[str] = None
        self._anim_x: Optional[np.ndarray] = None
        self._anim_y: Optional[np.ndarray] = None
        self._anim_u: Optional[np.ndarray] = None   # shape: (Nt, Ny, Nx)
        self._anim_t: Optional[np.ndarray] = None   # shape: (Nt,)

        main_layout = QHBoxLayout(self)

        # ---------- 左：run 一覧 ----------
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Heat2D Runs"))

        self.run_list = QListWidget()
        self.run_list.setMaximumWidth(220)
        self.run_list.currentItemChanged.connect(self.on_run_selected)
        left_layout.addWidget(self.run_list, 1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_runs)
        left_layout.addWidget(self.refresh_button)

        main_layout.addLayout(left_layout, 1)

        # ---------- 右：Config / Heatmap ----------
        right_layout = QVBoxLayout()

        right_layout.addWidget(QLabel("Config / Summary"))
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        right_layout.addWidget(self.config_text, 2)

        right_layout.addWidget(QLabel("Solution / Animation (Heatmap)"))
        self.canvas = Heat2DCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(self.canvas, 4)

        # --- アニメーション用スライダー ---
        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: - / -")
        slider_layout.addWidget(self.frame_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        slider_layout.addWidget(self.frame_slider, 1)

        right_layout.addLayout(slider_layout)

        main_layout.addLayout(right_layout, 3)

        # 初回一覧読み込み
        self.refresh_runs()

    # ========= run 一覧読み込み =========

    def refresh_runs(self) -> None:
        self.run_list.clear()
        runs = log_utils.list_runs()
        if not runs:
            self.run_list.addItem("(no runs)")
            self.run_list.setEnabled(False)
            self._reset_anim_state()
            return

        self.run_list.setEnabled(True)
        for info in runs:
            self.run_list.addItem(info.run_id)

    # ========= run 選択時 =========

    @Slot("QListWidgetItem*", "QListWidgetItem*")
    def on_run_selected(self, current, previous) -> None:
        if current is None:
            return
        run_id = current.text()
        if run_id == "(no runs)":
            return

        self._current_run_id = run_id

        # ---- Config / Summary 表示 ----
        summary = log_utils.load_summary(run_id)
        if summary is not None:
            self.config_text.setPlainText(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
        else:
            cfg = log_utils.load_config(run_id)
            if cfg is not None:
                self.config_text.setPlainText(
                    json.dumps(cfg, indent=2, ensure_ascii=False)
                )
            else:
                self.config_text.setPlainText("(no config.json / summary.json)")

        # ---- 静的な T_final ヒートマップ ----
        self._show_static_heatmap(run_id)

        # ---- アニメーションデータ読み込み & スライダー更新 ----
        self._load_animation_data(run_id)

        self.runSelected.emit(run_id)

    # ========= 静的ヒートマップ描画 =========

    def _show_static_heatmap(self, run_id: str) -> None:
        ev = log_utils.load_eval(run_id)
        x = y = u = None

        if ev is not None and all(k in ev for k in ("x", "y", "u")):
            x = np.asarray(ev["x"], dtype=float)
            y = np.asarray(ev["y"], dtype=float)
            u = np.asarray(ev["u"], dtype=float)
        else:
            summary = log_utils.load_summary(run_id)
            if summary is not None:
                sol = summary.get("results", {}).get("solution_2d", {})
                if all(k in sol for k in ("x", "y", "u")):
                    x = np.asarray(sol["x"], dtype=float)
                    y = np.asarray(sol["y"], dtype=float)
                    u = np.asarray(sol["u"], dtype=float)

        if x is None or y is None or u is None:
            ax = self.canvas.ax
            ax.clear()
            ax.text(0.5, 0.5, "No 2D solution data",
                    ha="center", va="center", color="white")
            self.canvas.apply_dark_style()
            self.canvas.draw_idle()
            return

        self.canvas.plot_solution(x, y, u)

    # ========= アニメーションデータ =========

    def _reset_anim_state(self) -> None:
        self._anim_x = None
        self._anim_y = None
        self._anim_u = None
        self._anim_t = None

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.blockSignals(False)

        self.frame_label.setText("Frame: - / -")

    def _load_animation_data(self, run_id: str) -> None:
        """
        *_anim.npz があれば読み込み、スライダーを有効化。
        なければスライダーは無効のまま。
        """
        self._reset_anim_state()

        # sim_log_utils 側に path_anim があればそれを優先
        anim_path = None
        if hasattr(log_utils, "path_anim"):
            anim_path = log_utils.path_anim(run_id)
        elif hasattr(log_utils, "LOG_DIR"):
            anim_path = log_utils.LOG_DIR / f"{run_id}_anim.npz"

        if anim_path is None or not anim_path.exists():
            return

        try:
            data = np.load(anim_path)
        except Exception:
            return

        try:
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)
            u_all = np.asarray(data["u"], dtype=float)
            t = data["t"] if "t" in data else None
        except KeyError:
            return

        if u_all.ndim != 3:
            return

        self._anim_x = x
        self._anim_y = y
        self._anim_u = u_all
        self._anim_t = np.asarray(t, dtype=float) if t is not None else None

        n_frames = u_all.shape[0]

        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(n_frames - 1)
        self.frame_slider.setValue(n_frames - 1)
        self.frame_slider.setEnabled(True)
        self.frame_slider.blockSignals(False)

        self.frame_label.setText(f"Frame: {n_frames} / {n_frames}")

    # ========= スライダー操作 =========

    @Slot(int)
    def on_frame_slider_changed(self, idx: int) -> None:
        if self._anim_u is None or self._anim_x is None or self._anim_y is None:
            return

        n_frames = self._anim_u.shape[0]
        if n_frames == 0:
            return

        idx_clamped = max(0, min(idx, n_frames - 1))
        u_frame = self._anim_u[idx_clamped]

        self.canvas.plot_solution(self._anim_x, self._anim_y, u_frame)

        if self._anim_t is not None and 0 <= idx_clamped < len(self._anim_t):
            t_val = float(self._anim_t[idx_clamped])
            self.canvas.ax.set_title(f"Heat2D: u(x, y, t={t_val:.3f})")
            self.canvas.draw_idle()

        self.frame_label.setText(
            f"Frame: {idx_clamped + 1} / {n_frames}"
        )


class Heat2DLogsPlugin(EditorPlugin):
    plugin_id = "heat2d_logs"
    display_name = "Heat2D Logs"

    def create_dock(self, main_window):
        widget = Heat2DLogsPage(main_window)

        if hasattr(main_window, "on_heat2d_log_selected"):
            widget.runSelected.connect(main_window.on_heat2d_log_selected)

        dock = QDockWidget(self.display_name, main_window)
        dock.setWidget(widget)
        dock.setObjectName(self.plugin_id)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
            | Qt.BottomDockWidgetArea
        )
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.hide()

        if hasattr(main_window, "view_menu"):
            main_window.view_menu.addAction(dock.toggleViewAction())

        return dock


register_builtin_plugin(Heat2DLogsPlugin)
