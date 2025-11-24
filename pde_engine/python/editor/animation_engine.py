from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from backend import log_utils
if TYPE_CHECKING:
    from .viewport_1d import ViewportWidget


@dataclass
class AnimationState:
    run_id: str
    backend: Optional[str]
    config: Dict[str, Any]
    device: Any
    model: Any
    x: Any  # torch.Tensor を想定
    T_final: float
    N_frames: int


class AnimationEngine:
    """
    1D PINN の時間発展アニメーションを担当するクラス。

    ViewportWidget からは
      - setup(run_id)
      - render_frame(step)
    を呼ぶだけにする。
    """

    def __init__(self, viewport: "ViewportWidget"):
        self.viewport = viewport
        self.state: Optional[AnimationState] = None

    # --------- ユーティリティ ---------

    def _draw_message(self, msg: str):
        ax = self.viewport.canvas.axes
        ax.clear()
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        self.viewport.canvas.apply_dark_style()
        self.viewport.canvas.draw()

    # --------- 準備（model / config 読み込み） ---------

    def setup(self, run_id: str) -> None:
        """
        run_id に対応する config / model を読み込み、
        アニメーション用の状態を構築する。
        初期フレーム（T_final 側）もここで描画する。
        """
        ax = self.viewport.canvas.axes
        ax.clear()

        config_path = log_utils.path_config(run_id)
        model_path = log_utils.path_model(run_id)

        if not config_path.exists() or not model_path.exists():
            self.state = None
            self.viewport.info_label.setText(
                f"{run_id}: no config/model for animation"
            )
            self._draw_message("No config/model for this run")
            return

        # torch が無い環境なら静的な eval 表示にフォールバック
        try:
            import torch  # type: ignore
        except ImportError:
            self.state = None
            self.viewport.info_label.setText(
                f"{run_id}: torch not available, fallback to static eval"
            )
            self._render_static_from_eval(run_id)
            return

        # config 読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # backend 判定（heat / wave）
        summary = log_utils.load_summary(run_id)
        backend = None
        if summary is not None:
            backend = summary.get("run", {}).get("backend", None)

        if backend == "pinn_wave1d":
            import pinn_wave1d as pinn_module  # type: ignore
        else:
            import pinn_heat1d as pinn_module  # type: ignore

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

        self.state = AnimationState(
            run_id=run_id,
            backend=backend,
            config=config,
            device=device,
            model=model,
            x=x,
            T_final=T_final,
            N_frames=N_frames,
        )

        # スライダ設定
        slider = self.viewport.time_slider
        slider.blockSignals(True)
        slider.setMinimum(0)
        slider.setMaximum(N_frames)
        slider.setValue(N_frames)
        slider.blockSignals(False)

        # 初期フレーム（T_final 側）を描画
        self.render_frame(N_frames)

    # --------- フォールバック描画（eval.json だけで） ---------

    def _render_static_from_eval(self, run_id: str) -> None:
        ev = log_utils.load_eval(run_id)
        ax = self.viewport.canvas.axes
        ax.clear()

        if ev is None:
            self._draw_message("No eval.json for this run")
            self.viewport.info_label.setText(f"Result: {run_id} (no eval)")
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

        self.viewport.info_label.setText(
            f"Animation (fallback): {run_id} | static T_final"
        )
        self.viewport.canvas.apply_dark_style()
        self.viewport.canvas.draw()

    # --------- フレーム描画 ---------

    def render_frame(self, step: int) -> None:
        """
        スライダ値 `step` に応じたフレームを描画する。
        state が無ければ eval.json ベースの静止図になる。
        """
        run_id = self.viewport.state.current_run_id
        ax = self.viewport.canvas.axes
        ax.clear()

        if not run_id:
            self._draw_message("No run selected")
            self.viewport.info_label.setText("No run yet")
            return

        state = self.state

        # 1) PINN モデルを用いたアニメーション
        if state is not None:
            import torch  # type: ignore

            N_frames = int(state.N_frames)
            T_final = float(state.T_final)
            step_clamped = max(0, min(int(step), N_frames))
            t = T_final * (step_clamped / max(1, N_frames))

            device = state.device
            model = state.model
            x = state.x

            with torch.no_grad():
                t_tensor = torch.full_like(x, fill_value=t, device=device)
                u_t = model(x, t_tensor).cpu().numpy().flatten()
            x_np = x.cpu().numpy().flatten()

            ax.plot(x_np, u_t, label="PINN", linewidth=2)
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.set_title(f"Animation (run_id={run_id}, t={t:.3f})")
            ax.legend()

            self.viewport.state.anim_step = int(step_clamped)
            self.viewport.info_label.setText(
                f"Animation: {run_id} | step {step_clamped}/{N_frames}  t={t:.3f}"
            )
            self.viewport.canvas.apply_dark_style()
            self.viewport.canvas.draw()
            return

        # 2) フォールバック（eval.json）
        self._render_static_from_eval(run_id)
