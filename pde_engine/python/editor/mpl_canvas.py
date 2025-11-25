# editor/mpl_canvas.py
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


DARK_BG = "#2b2b2b"
FG_COLOR = "white"


class MatplotlibCanvas(FigureCanvasQTAgg):
    """
    1D グラフ用の共通 Matplotlib キャンバス。
    Viewport / Logs などで再利用する。
    """
    def __init__(self, parent=None, width: float = 5, height: float = 3, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.apply_dark_style()

    def apply_dark_style(self):
        ax = self.axes

        # 背景
        self.fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)

        # 軸・ラベル
        ax.tick_params(axis="both", colors=FG_COLOR)
        ax.xaxis.label.set_color(FG_COLOR)
        ax.yaxis.label.set_color(FG_COLOR)

        # 枠線
        for spine in ax.spines.values():
            spine.set_color(FG_COLOR)

        # タイトル
        ax.title.set_color(FG_COLOR)
