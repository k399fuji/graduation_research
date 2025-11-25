# editor/viewport_modes.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np

from backend import log_utils

if TYPE_CHECKING:
    from .viewport_1d import ViewportWidget

LOSS_LINEWIDTH = 0.5


class ViewMode(ABC):
    """
    Viewport の表示モードの共通インターフェース。
    """

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
        if ev is None:
            ax.text(0.5, 0.5, "No eval.json for this run",
                    ha="center", va="center")
            vp.info_label.setText(f"Result: {run_id} (no eval)")
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

        vp.info_label.setText(f"Result: {run_id}  |  mode: Solution")
        msg = f"Result: {run_id}  |  mode: Solution"
        vp.state.info_message = msg
        vp.info_label.setText(msg)


class ErrorMode(ViewMode):
    def __init__(self, viewport: "ViewportWidget"):
        super().__init__(viewport, "Error")

    def render(self, run_id: str) -> None:
        vp = self.viewport
        ax = vp.canvas.axes

        ev = vp._load_eval(run_id)
        if ev is None:
            ax.text(0.5, 0.5, "No eval.json for this run",
                    ha="center", va="center")
            vp.info_label.setText(f"Result: {run_id} (no eval)")
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

        vp.info_label.setText(f"Result: {run_id}  |  mode: Error")


class LossMode(ViewMode):
    def __init__(self, viewport: "ViewportWidget"):
        super().__init__(viewport, "Loss")

    def render(self, run_id: str) -> None:
        vp = self.viewport
        ax = vp.canvas.axes

        rows = log_utils.load_loss_csv(run_id)
        if not rows:
            ax.text(0.5, 0.5, "No CSV loss log", ha="center", va="center")
            vp.info_label.setText(f"Result: {run_id} (no CSV)")
            return

        epochs = [int(r["epoch"]) for r in rows]

        def get_series(col_name: str, fallback: Optional[str] = None):
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

        vp.info_label.setText(f"Result: {run_id}  |  mode: Loss")


class AnimationMode(ViewMode):
    def __init__(self, viewport: "ViewportWidget"):
        super().__init__(viewport, "Animation")

    def render(self, run_id: str) -> None:
        """
        アニメーションモードでは、viewport 側の AnimationEngine に
        セットアップを依頼するだけ。描画処理は AnimationEngine に委譲。
        """
        vp = self.viewport
        vp.animation.setup(run_id)
        # 初期フレーム描画は animation.setup 内で行われる
