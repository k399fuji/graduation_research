# editor/heat2d_tab.py
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox,
    QLineEdit, QPushButton, QTextEdit, QApplication,
    QSplitter, QToolButton,
)

from backend.heat2d_backend import Heat2DConfig, Heat2DResult, run_heat2d
from editor.viewmodels import Heat2DState

try:
    import heat2d_cpp  # noqa: F401
except ImportError:
    heat2d_cpp = None

DARK_BG = "#2b2b2b"
FG_COLOR = "white"


# editor/heat2d_tab.py の先頭付近はそのまま

class Heat2DCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.im = None
        self.cbar = None
        self.apply_dark_style()

    def apply_dark_style(self):
        self.fig.patch.set_facecolor(DARK_BG)
        self.ax.set_facecolor(DARK_BG)

        for spine in self.ax.spines.values():
            spine.set_color(FG_COLOR)
        self.ax.tick_params(axis="both", colors=FG_COLOR)
        self.ax.xaxis.label.set_color(FG_COLOR)
        self.ax.yaxis.label.set_color(FG_COLOR)
        self.ax.title.set_color(FG_COLOR)

        if self.cbar is not None:
            self.cbar.outline.set_edgecolor(FG_COLOR)
            self.cbar.ax.tick_params(colors=FG_COLOR)
            for label in self.cbar.ax.get_yticklabels():
                label.set_color(FG_COLOR)

    def clear_canvas(self):
        """ヒートマップを描き直す前に Figure 全体を安全に初期化する。"""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.im = None
        self.cbar = None
        self.apply_dark_style()

    def plot_solution(self, x, y, u):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u.reshape(len(y), len(x))

        # ★ 以前の self.ax.clear() / self.cbar.remove() の代わりに
        #   Figure ごと綺麗に作り直す
        self.clear_canvas()

        extent = [x.min(), x.max(), y.min(), y.max()]
        self.im = self.ax.imshow(
            u,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Heat2D: u(x, y, T_final)")

        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.apply_dark_style()
        self.draw_idle()


class Heat2DTab(QWidget):
    """
    2D Heat 方程式の数値シミュレーションタブ。
    左：パラメータ＋ログ / 右：ヒートマップ（QSplitter）
    上部のボタンで左右どちらも折りたたみ可能。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.state = Heat2DState()

        # 折りたたみ前のサイズ記憶用
        self._last_left_size = 350
        self._last_right_size = 850

        # ===== ルートレイアウト =====
        root_layout = QVBoxLayout(self)

        # ---- 上部ツールバー（常に見えるトグルボタン）----
        toolbar_layout = QHBoxLayout()
        title_label = QLabel("Heat2D Simulation")
        toolbar_layout.addWidget(title_label)
        toolbar_layout.addStretch()

        self.btn_toggle_left = QToolButton()
        self.btn_toggle_left.setText("Hide Params/Log")
        self.btn_toggle_left.setCheckable(True)
        self.btn_toggle_left.toggled.connect(self.on_toggle_left_panel)
        toolbar_layout.addWidget(self.btn_toggle_left)

        self.btn_toggle_right = QToolButton()
        self.btn_toggle_right.setText("Hide Heatmap")
        self.btn_toggle_right.setCheckable(True)
        self.btn_toggle_right.toggled.connect(self.on_toggle_right_panel)
        toolbar_layout.addWidget(self.btn_toggle_right)

        root_layout.addLayout(toolbar_layout)

        # ---- Splitter（左：パネル / 右：キャンバス）----
        self.splitter = QSplitter(Qt.Horizontal, self)
        root_layout.addWidget(self.splitter)

        # ===== 左パネル（パラメータ＋ログ） =====
        self.left_panel = QWidget(self)
        left_layout = QVBoxLayout(self.left_panel)

        left_layout.addWidget(QLabel("Heat2D Parameters / Log"))

        # パラメータフォーム
        form_layout = QFormLayout()

        self.spin_Lx = QDoubleSpinBox()
        self.spin_Lx.setRange(0.1, 10.0)
        self.spin_Lx.setSingleStep(0.1)
        self.spin_Lx.setValue(1.0)

        self.spin_Ly = QDoubleSpinBox()
        self.spin_Ly.setRange(0.1, 10.0)
        self.spin_Ly.setSingleStep(0.1)
        self.spin_Ly.setValue(1.0)

        self.spin_Nx = QSpinBox()
        self.spin_Nx.setRange(10, 301)
        self.spin_Nx.setSingleStep(10)
        self.spin_Nx.setValue(101)

        self.spin_Ny = QSpinBox()
        self.spin_Ny.setRange(10, 301)
        self.spin_Ny.setSingleStep(10)
        self.spin_Ny.setValue(101)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setDecimals(5)
        self.spin_alpha.setRange(1e-5, 1.0)
        self.spin_alpha.setSingleStep(0.001)
        self.spin_alpha.setValue(0.01)

        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setDecimals(6)
        self.spin_dt.setRange(1e-6, 1.0)
        self.spin_dt.setSingleStep(1e-4)
        self.spin_dt.setValue(1e-4)

        self.spin_Tfinal = QDoubleSpinBox()
        self.spin_Tfinal.setRange(0.001, 10.0)
        self.spin_Tfinal.setSingleStep(0.1)
        self.spin_Tfinal.setValue(0.1)

        self.combo_ic_type = QComboBox()
        self.combo_ic_type.addItems(["gaussian", "sinexy"])

        self.spin_gaussian_kx = QDoubleSpinBox()
        self.spin_gaussian_kx.setRange(1.0, 500.0)
        self.spin_gaussian_kx.setSingleStep(10.0)
        self.spin_gaussian_kx.setValue(100.0)

        self.spin_gaussian_ky = QDoubleSpinBox()
        self.spin_gaussian_ky.setRange(1.0, 500.0)
        self.spin_gaussian_ky.setSingleStep(10.0)
        self.spin_gaussian_ky.setValue(100.0)

        self.edit_tag = QLineEdit("heat2d")

        form_layout.addRow("Nx (grid in x)", self.spin_Nx)
        form_layout.addRow("Ny (grid in y)", self.spin_Ny)
        form_layout.addRow("Lx (length in x)", self.spin_Lx)
        form_layout.addRow("Ly (length in y)", self.spin_Ly)
        form_layout.addRow("alpha (diffusivity)", self.spin_alpha)
        form_layout.addRow("dt (time step)", self.spin_dt)
        form_layout.addRow("T_final", self.spin_Tfinal)
        form_layout.addRow("IC type", self.combo_ic_type)
        form_layout.addRow("Gaussian kx", self.spin_gaussian_kx)
        form_layout.addRow("Gaussian ky", self.spin_gaussian_ky)
        form_layout.addRow("tag", self.edit_tag)

        left_layout.addLayout(form_layout)

        # 実行ボタン行
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run 2D Heat Simulation")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        self.run_id_label = QLabel("run_id: (not run yet)")
        button_layout.addWidget(self.run_id_label)

        self.info_label = QLabel("Ready.")
        button_layout.addWidget(self.info_label)

        button_layout.addStretch()
        left_layout.addLayout(button_layout)

        # ログエリア
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(QLabel("Log:"))
        left_layout.addWidget(self.log_text)

        self.splitter.addWidget(self.left_panel)

        # ===== 右パネル（キャンバス） =====
        self.canvas = Heat2DCanvas(self, width=5, height=4, dpi=100)
        self.splitter.addWidget(self.canvas)

        # 初期の幅配分（左:右 = 1:2 くらい）
        self.splitter.setSizes([self._last_left_size, self._last_right_size])

    # ---------------- ユーティリティ ----------------

    def append_log(self, text: str):
        self.log_text.append(text)

    def get_config(self) -> dict:
        """
        UI から設定値を収集し、Heat2DState と backend 用 config dict の両方に使える
        標準化された辞書を返す。
        """
        cfg = {
            "Nx_cpp": int(self.spin_Nx.value()),
            "Ny_cpp": int(self.spin_Ny.value()),
            "Lx": float(self.spin_Lx.value()),
            "Ly": float(self.spin_Ly.value()),
            "alpha": float(self.spin_alpha.value()),
            "dt_cpp": float(self.spin_dt.value()),
            "T_final": float(self.spin_Tfinal.value()),
            "ic_type": self.combo_ic_type.currentText(),
            "gaussian_kx": float(self.spin_gaussian_kx.value()),
            "gaussian_ky": float(self.spin_gaussian_ky.value()),
            "tag": self.edit_tag.text().strip() or "heat2d",
            "solver_type": "heat2d",
        }

        # ViewModel にも反映（型の都合で ignore）
        setattr(self.state, "config", cfg)

        return cfg

    # ---------------- 折りたたみ系スロット ----------------

    @Slot(bool)
    def on_toggle_left_panel(self, checked: bool):
        sizes = self.splitter.sizes()
        total = sum(sizes) or (self._last_left_size + self._last_right_size)

        if checked:
            # 折りたたむ前のサイズを保存
            if sizes[0] > 0:
                self._last_left_size = sizes[0]
            # 左を 0、右を全てに
            self.left_panel.hide()
            self.splitter.setSizes([0, total])
            self.btn_toggle_left.setText("Show Params/Log")

            # 両方 0 にならないように保険
            if self.btn_toggle_right.isChecked() and total == 0:
                self.btn_toggle_right.setChecked(False)
        else:
            # 復元
            self.left_panel.show()
            left = self._last_left_size or int(total * 0.3)
            right = total - left
            if right <= 0:
                right = int(total * 0.7)
            self.splitter.setSizes([left, right])
            self.btn_toggle_left.setText("Hide Params/Log")

    @Slot(bool)
    def on_toggle_right_panel(self, checked: bool):
        sizes = self.splitter.sizes()
        total = sum(sizes) or (self._last_left_size + self._last_right_size)

        if checked:
            if sizes[1] > 0:
                self._last_right_size = sizes[1]
            # 右を 0、左を全てに
            self.canvas.hide()
            self.splitter.setSizes([total, 0])
            self.btn_toggle_right.setText("Show Heatmap")

            # 両方 0 にならないように保険
            if self.btn_toggle_left.isChecked() and total == 0:
                self.btn_toggle_left.setChecked(False)
        else:
            self.canvas.show()
            right = self._last_right_size or int(total * 0.7)
            left = total - right
            if left <= 0:
                left = int(total * 0.3)
            self.splitter.setSizes([left, right])
            self.btn_toggle_right.setText("Hide Heatmap")

    # ---------------- 実行ボタン ----------------

    @Slot()
    def on_run_clicked(self):
        # Config を取得＆ state 更新
        cfg_dict = self.get_config()

        self.state.is_running = True
        self.state.last_error = None
        self.state.info_message = "Running 2D Heat simulation..."

        self.info_label.setText(self.state.info_message)
        self.run_button.setEnabled(False)
        QApplication.processEvents()

        cfg = Heat2DConfig(
            Nx_cpp=cfg_dict["Nx_cpp"],
            Ny_cpp=cfg_dict["Ny_cpp"],
            Lx=cfg_dict["Lx"],
            Ly=cfg_dict["Ly"],
            alpha=cfg_dict["alpha"],
            dt_cpp=cfg_dict["dt_cpp"],
            T_final=cfg_dict["T_final"],
            ic_type=cfg_dict["ic_type"].lower(),
            gaussian_kx=cfg_dict["gaussian_kx"],
            gaussian_ky=cfg_dict["gaussian_ky"],
            tag=cfg_dict["tag"],
        )

        self.append_log(f"Config: {cfg}")

        try:
            result: Heat2DResult = run_heat2d(cfg)
        except Exception as e:
            # 失敗時
            self.state.is_running = False
            self.state.last_error = repr(e)
            self.state.info_message = "Error during Heat2D simulation."

            self.append_log(f"Error: {e!r}")
            self.info_label.setText(self.state.info_message)
            self.run_button.setEnabled(True)
            return

        # 成功時：state 更新
        self.state.is_running = False
        self.state.last_run_id = result.run_id
        self.state.info_message = "Done."

        # 描画
        self.canvas.plot_solution(result.x, result.y, result.u)

        # ラベル・ログ更新
        self.run_id_label.setText(f"run_id: {self.state.last_run_id}")
        self.info_label.setText(self.state.info_message)

        if result.config_json_path is not None:
            self.append_log(f"Config JSON: {result.config_json_path}")
        if result.eval_json_path is not None:
            self.append_log(f"Eval JSON: {result.eval_json_path}")
        if result.summary_json_path is not None:
            self.append_log(f"Summary JSON: {result.summary_json_path}")

        self.run_button.setEnabled(True)
