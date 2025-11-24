# editor/viewport_1d.py
from __future__ import annotations

import json
from typing import Optional, Dict, Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider,
)

from backend import log_utils


LOSS_LINEWIDTH = 0.5


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

        self._current_run_id: Optional[str] = None
        self._current_anim_step: int = 0
        self._anim_state: Optional[Dict[str, Any]] = None

        layout = QVBoxLayout(self)

        # --- 上部: モード切替 ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("View:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Solution", "Error", "Loss", "Animation"])
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

    # ========= Animation 用セットアップ =========

    def _setup_animation_state(self, run_id: str, ax):
        import torch

        config_path = log_utils.path_config(run_id)
        model_path = log_utils.path_model(run_id)

        if not config_path.exists() or not model_path.exists():
            ax.text(0.5, 0.5, "No config/model for this run",
                    ha="center", va="center")
            self.info_label.setText(f"{run_id}: no model checkpoint")
            self._anim_state = None
            return

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        summary = log_utils.load_summary(run_id)
        backend = None
        if summary is not None:
            backend = summary.get("run", {}).get("backend", None)

        if backend == "pinn_wave1d":
            import pinn_wave1d as pinn_module
        else:
            import pinn_heat1d as pinn_module

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = pinn_module.build_model(config, device)

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        L = float(config.get("L", 1.0))
        T_final = float(config.get("T_final", 1.0))
        Nx_eval = int(config.get("Nx_eval", 201))

        x = torch.linspace(0.0, L, Nx_eval, device=device).view(-1, 1)
        N_frames = 100

        self._anim_state = {
            "run_id": run_id,
            "backend": backend,
            "config": config,
            "device": device,
            "model": model,
            "x": x,
            "T_final": T_final,
            "N_frames": N_frames,
        }

        self.time_slider.blockSignals(True)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(N_frames)
        self.time_slider.setValue(N_frames)
        self.time_slider.blockSignals(False)

        self._update_animation_frame(N_frames, ax)

    # ========= 外部 API =========

    def show_result(self, run_id: str):
        self._current_run_id = run_id
        self._update_plot()

    # ========= スロット =========

    @Slot()
    def on_mode_changed(self):
        if self._current_run_id is not None:
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
        self._setup_animation_state(run_id, ax)

    # ========= メイン描画更新 =========

    def _update_plot(self):
        ax = self.canvas.axes
        ax.clear()

        run_id = self._current_run_id
        if not run_id:
            ax.text(0.5, 0.5, "No run selected", ha="center", va="center")
            self.info_label.setText("No run yet")
            self.canvas.draw()
            return

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
            self.info_label.setText(f"{dim}D run: {run_id}  (view in 2D Heat tab)")
            self.canvas.draw()
            return

        mode = self.mode_combo.currentText()
        self.time_slider.setVisible(mode == "Animation")

        ev = None
        if mode in ("Solution", "Error"):
            ev = self._load_eval(run_id)

        if mode == "Solution":
            if ev is None or any(k not in ev for k in ("x_pinn", "u_pinn", "x_cpp", "u_cpp")):
                ax.text(
                    0.5, 0.5,
                    "This run has no 1D eval data.\n(Heat2D etc. are not shown here.)",
                    ha="center", va="center"
                )
                self.info_label.setText(f"Result: {run_id}  |  mode: {mode} (unsupported)")
                self.canvas.draw()
                return
            self._plot_solution(ax, run_id, ev)

        elif mode == "Error":
            self._plot_error(ax, run_id, ev)

        elif mode == "Loss":
            self._plot_loss(ax, run_id)

        elif mode == "Animation":
            self._plot_animation_or_fallback(ax, run_id)
            self.canvas.apply_dark_style()
            self.canvas.draw()
            return

        else:
            ax.text(0.5, 0.5, f"Unknown mode: {mode}",
                    ha="center", va="center")
            self.info_label.setText(f"Unknown mode: {mode}")

        self.canvas.apply_dark_style()
        self.canvas.draw()

    def _update_animation_frame(self, step: int, ax=None):
        if ax is None:
            ax = self.canvas.axes
        ax.clear()

        run_id = self._current_run_id
        if not run_id:
            ax.text(0.5, 0.5, "No run selected", ha="center", va="center")
            self.info_label.setText("No run yet")
            self.canvas.apply_dark_style()
            self.canvas.draw()
            return

        state = self._anim_state

        # --- 1) PINN モデルを使ったアニメーション ---
        if state is not None:
            import torch

            N_frames = int(state.get("N_frames", 100))
            T_final = float(state.get("T_final", 1.0))
            step_clamped = max(0, min(int(step), N_frames))
            t = T_final * (step_clamped / max(1, N_frames))

            device = state["device"]
            model = state["model"]
            x = state["x"]

            with torch.no_grad():
                t_tensor = torch.full_like(x, fill_value=t, device=device)
                u_t = model(x, t_tensor).cpu().numpy().flatten()
            x_np = x.cpu().numpy().flatten()

            ax.plot(x_np, u_t, label="PINN", linewidth=2)
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.set_title(f"Animation (run_id={run_id}, t={t:.3f})")
            ax.legend()

            self._current_anim_step = int(step_clamped)
            self.info_label.setText(
                f"Animation: {run_id}  |  step {step_clamped}/{N_frames}  t={t:.3f}"
            )
            self.canvas.apply_dark_style()
            self.canvas.draw()
            return

        # --- 2) フォールバック: eval.json の T_final ---
        ev = self._load_eval(run_id)
        if ev is None:
            ax.text(0.5, 0.5, "No eval.json for this run",
                    ha="center", va="center")
            self.info_label.setText(f"Result: {run_id} (no eval)")
            self.canvas.apply_dark_style()
            self.canvas.draw()
            return

        x_pinn = np.asarray(ev.get("x_pinn"), dtype=float)
        u_pinn = np.asarray(ev.get("u_pinn"), dtype=float)
        x_cpp = np.asarray(ev.get("x_cpp"), dtype=float)
        u_cpp = np.asarray(ev.get("u_cpp"), dtype=float)

        ax.plot(x_pinn, u_pinn, label="PINN (T_final)", linewidth=2)
        if len(x_cpp) == len(u_cpp):
            ax.plot(x_cpp, u_cpp, "--", label="C++ reference (T_final)")

        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"Animation (run_id={run_id}, static T_final)")
        ax.legend()

        self._current_anim_step = int(step)
        self.info_label.setText(
            f"Animation (fallback): {run_id}  |  step={step} (static T_final)"
        )
        self.canvas.apply_dark_style()
        self.canvas.draw()

    @Slot(int)
    def on_time_slider_changed(self, value: int):
        if self.mode_combo.currentText() != "Animation":
            return
        ax = self.canvas.axes
        self._update_animation_frame(value, ax)
