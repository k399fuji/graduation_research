# editor/viewport_modes.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .viewport_1d import ViewportWidget


class ViewportMode:
    """
    Viewport の表示モードを表す Strategy 基底クラス。
    （実装は各サブクラスに任せる）
    """
    name: str = ""

    def __init__(self, viewport: "ViewportWidget") -> None:
        self.viewport = viewport

    def render(self, run_id: str) -> None:
        raise NotImplementedError
        

class SolutionMode(ViewportMode):
    name = "Solution"

    def render(self, run_id: str) -> None:
        ax = self.viewport.canvas.axes
        # eval.json のロードは既存のヘルパをそのまま利用
        ev = self.viewport._load_eval(run_id)
        # 実際の描画ロジックも既存メソッドを呼び出すだけ
        self.viewport._plot_solution(ax, run_id, ev)


class ErrorMode(ViewportMode):
    name = "Error"

    def render(self, run_id: str) -> None:
        ax = self.viewport.canvas.axes
        ev = self.viewport._load_eval(run_id)
        self.viewport._plot_error(ax, run_id, ev)


class LossMode(ViewportMode):
    name = "Loss"

    def render(self, run_id: str) -> None:
        ax = self.viewport.canvas.axes
        self.viewport._plot_loss(ax, run_id)


class AnimationMode(ViewportMode):
    name = "Animation"

    def render(self, run_id: str) -> None:
        """
        現時点では既存の _plot_animation_or_fallback を呼び出すだけ。
        Step B で AnimationEngine 側に中身を移す予定。
        """
        ax = self.viewport.canvas.axes
        self.viewport._plot_animation_or_fallback(ax, run_id)
