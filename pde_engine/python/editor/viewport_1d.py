# editor/viewport_1d.py
from __future__ import annotations
import json
from typing import Optional, Dict, Any, TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider,
)

from backend import log_utils
from abc import ABC, abstractmethod
from editor.animation_engine import AnimationEngine
from editor.viewmodels import ViewportState 

LOSS_LINEWIDTH = 0.5

if TYPE_CHECKING:
    from .viewport_1d import ViewportWidget 

class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.apply_dark_style()

    def apply_dark_style(self):
        ax = self.axes
        self.fig.patch.set_facecolor("#2b2b2b")
        ax.set_facecolor("#2b2b2b")

        ax.tick_params(axis="both", colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.title.set_color("white")


class ViewMode(ABC):
    def __init__(self, viewport: "ViewportWidget", name: str):
        self.viewport = viewport
        self.name = name

    @abstractmethod
    def render(self, run_id: str) -> None:
        """指定 run_id を描画する"""
        raise NotImplementedError


class SolutionMode(ViewMode):
    def __init__(self, viewport: "ViewportWidget"):
        super().__init__(viewport, "Solution")

    def render(self, run_id: str) -> None:
        vp = self.viewport
        ax = vp.canvas.axes
        ev = vp._load_eval(run_id)
        vp._plot_solution(ax, run_id, ev)


class ErrorMode(ViewMode):
    def __init__(self, viewport: "ViewportWidget"):
        super().__init__(viewport, "Error")

    def render(self, run_id: str) -> None:
        vp = self.viewport
        ax = vp.canvas.axes
        ev = vp._load_eval(run_id)
        vp._plot_error(ax, run_id, ev)


class LossMode(ViewMode):
    def __init__(self, viewport: "ViewportWidget"):
        super().__init__(viewport, "Loss")

    def render(self, run_id: str) -> None:
        vp = self.viewport
        ax = vp.canvas.axes
        vp._plot_loss(ax, run_id)


class AnimationMode(ViewMode):
    def __init__(self, viewport: "ViewportWidget"):
        super().__init__(viewport, "Animation")

    def render(self, run_id: str) -> None:
        vp = self.viewport
        ax = vp.canvas.axes
        vp._plot_animation_or_fallback(ax, run_id)

class ViewportWidget(QWidget):
    """
    中央ビュー。1D の run のみを表示する。
    モード:
      - Solution
      - Error
      - Loss
      - Animation
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.state = ViewportState()

        layout = QVBoxLayout(self)

        # --- 上部: モード切替 ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("View:"))

        self.mode_combo = QComboBox()
        self._init_modes()
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # --- 中央: キャンバス ---
        self.canvas = MatplotlibCanvas(self, width=6, height=4, dpi=100)
        layout.addWidget(self.canvas, 1)

        # --- 下: アニメーション用スライダ ---
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.setValue(100)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.time_slider.setVisible(False)
        layout.addWidget(self.time_slider)

        # --- 下: 情報ラベル ---
        self.info_label = QLabel("No run yet")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # アニメーション専用エンジン
        self.animation = AnimationEngine(self)

        # 表示モードを登録
        self._init_modes()
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

    # ========= 共通ヘルパ =========

    def _load_eval(self, run_id: str) -> Optional[dict]:
        eval_path = log_utils.path_eval(run_id)
        if not eval_path.exists():
            return None
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _init_modes(self):
        """
        利用可能な表示モードを登録し、コンボボックスに反映する。
        """

        self._modes: dict[str, ViewMode] = {}

        self.mode_combo.clear()
        # Viewport 自身を渡してモードを生成
        for cls in (SolutionMode, ErrorMode, LossMode, AnimationMode):
            mode = cls(self)
            self._modes[mode.name] = mode
            self.mode_combo.addItem(mode.name)

        self.state.current_mode = "Solution"

    # ========= 外部 API =========

    def show_result(self, run_id: str):
        self.state.current_run_id = run_id
        self._update_plot()

    # ========= スロット =========

    @Slot()
    def on_mode_changed(self):
        self.state.current_mode = self.mode_combo.currentText()
        if self.state.current_run_id is not None:
            self._update_plot()

    # ========= モード別描画 =========

    def _plot_solution(self, ax, run_id: str, ev: Optional[dict]):
        if ev is None:
            ax.text(0.5, 0.5, "No eval.json for this run",
                    ha="center", va="center")
            self.info_label.setText(f"Result: {run_id} (no eval)")
            return

        x_pinn = np.asarray(ev.get("x_pinn"), dtype=float)
        u_pinn = np.asarray(ev.get("u_pinn"), dtype=float)
        x_cpp = np.asarray(ev.get("x_cpp"), dtype=float)
        u_cpp = np.asarray(ev.get("u_cpp"), dtype=float)

        ax.plot(x_pinn, u_pinn, label="PINN", linewidth=2)
        if len(x_cpp) == len(u_cpp):
            ax.plot(x_cpp, u_cpp, "--", label="C++ reference")

        ax.set_xlabel("x")
        ax.set_ylabel("u(x, T)")
        ax.set_title(f"Solution (run_id={run_id})")
        ax.legend()
        self.info_label.setText(f"Result: {run_id}  |  mode: Solution")

    def _plot_error(self, ax, run_id: str, ev: Optional[dict]):
        if ev is None:
            ax.text(0.5, 0.5, "No eval.json for this run",
                    ha="center", va="center")
            self.info_label.setText(f"Result: {run_id} (no eval)")
            return

        x_pinn = np.asarray(ev.get("x_pinn"), dtype=float)
        u_pinn = np.asarray(ev.get("u_pinn"), dtype=float)
        x_cpp = np.asarray(ev.get("x_cpp"), dtype=float)
        u_cpp = np.asarray(ev.get("u_cpp"), dtype=float)

        u_cpp_interp = np.interp(x_pinn, x_cpp, u_cpp)
        err = np.abs(u_pinn - u_cpp_interp)

        ax.plot(x_pinn, err, label="|u_pinn - u_cpp|")
        ax.set_xlabel("x")
        ax.set_ylabel("error")
        ax.set_yscale("log")
        ax.set_title(f"Error profile (run_id={run_id})")
        ax.legend()
        self.info_label.setText(f"Result: {run_id}  |  mode: Error")

    def _plot_loss(self, ax, run_id: str):
        rows = log_utils.load_loss_csv(run_id)
        if not rows:
            ax.text(0.5, 0.5, "No CSV loss log", ha="center", va="center")
            self.info_label.setText(f"Result: {run_id} (no CSV)")
            return

        epochs = [int(r["epoch"]) for r in rows]

        def get_series(col_name: str, fallback: str | None = None):
            if col_name in rows[0]:
                return [float(r[col_name]) for r in rows]
            if fallback and fallback in rows[0]:
                return [float(r[fallback]) for r in rows]
            return None

        loss_total = get_series("loss_total", "loss")
        loss_pde = get_series("loss_pde")
        loss_ic = get_series("loss_ic")
        loss_bc = get_series("loss_bc")

        if loss_total is not None:
            ax.plot(epochs, loss_total, label="loss_total",
                    linewidth=LOSS_LINEWIDTH)
        if loss_pde is not None:
            ax.plot(epochs, loss_pde, label="loss_pde",
                    linewidth=LOSS_LINEWIDTH)
        if loss_ic is not None:
            ax.plot(epochs, loss_ic, label="loss_ic",
                    linewidth=LOSS_LINEWIDTH)
        if loss_bc is not None:
            ax.plot(epochs, loss_bc, label="loss_bc",
                    linewidth=LOSS_LINEWIDTH)

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.set_title(f"Loss curves (run_id={run_id})")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        self.canvas.fig.subplots_adjust(right=0.78)
        self.info_label.setText(f"Result: {run_id}  |  mode: Loss")

    def _plot_animation_or_fallback(self, ax, run_id: str):
        self.animation.setup(run_id)

    # ========= メイン描画更新 =========

    def _update_plot(self):
        """現在の run_id とモードに基づいてグラフを更新する。"""

        ax = self.canvas.axes
        ax.clear()

        run_id = self.state.current_run_id
        if not run_id:
            ax.text(0.5, 0.5, "No run selected", ha="center", va="center")
            self.info_label.setText("No run yet")
            self.canvas.apply_dark_style()
            self.canvas.draw()
            return

        # 次元チェック（2D 以上はここでは表示しない）
        summary = log_utils.load_summary(run_id)
        dim = 1
        if summary is not None:
            dim = int(summary.get("problem", {}).get("dim", 1))

        if dim != 1:
            self.time_slider.setVisible(False)
            ax.text(
                0.5,
                0.5,
                f"{dim}D run ({run_id})\n"
                "Central viewer does not support this.\n"
                "Please use the dedicated 2D Heat tab.",
                ha="center",
                va="center",
            )
            self.info_label.setText(
                f"{dim}D run: {run_id} (view in 2D Heat tab)"
            )
            self.canvas.apply_dark_style()
            self.canvas.draw()
            return

        # ---- モードに応じて Strategy に委譲 ----
        mode_name = self.state.current_mode
        self.time_slider.setVisible(mode_name == "Animation")

        mode = self._modes.get(mode_name)
        if mode is None:
            ax.text(0.5, 0.5, f"Unknown mode: {mode_name}",
                    ha="center", va="center")
            self.info_label.setText(f"Unknown mode: {mode_name}")
        else:
            mode.render(run_id)

        self.canvas.apply_dark_style()
        self.canvas.draw()

    @Slot(int)
    def on_time_slider_changed(self, value: int):
        if self.mode_combo.currentText() != "Animation":
            return
        self.animation.render_frame(value)
