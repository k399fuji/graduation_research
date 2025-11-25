# editor/heat2d_tab.py
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Slot, Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QApplication,
    QSplitter,
)

from backend.heat2d_backend import Heat2DConfig, Heat2DResult, run_heat2d
from editor.viewmodels import Heat2DState

DARK_BG = "#2b2b2b"
FG_COLOR = "white"


# ============================================
# 2D 描画用キャンバス
# ============================================

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

    def plot_solution(self, x, y, u):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u.reshape(len(y), len(x))

        self.ax.clear()
        self.apply_dark_style()

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

        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.apply_dark_style()
        self.draw_idle()


# ============================================
# Viewer（右側：可視化専用）
# ============================================

class Heat2DViewer(QWidget):
    """
    Heat2D の結果を表示する Viewer。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.canvas = Heat2DCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas, 1)

        self.info_label = QLabel("Heat2D Viewer: no result yet")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

    def show_result(self, result: Heat2DResult) -> None:
        """
        Heat2DResult を受け取ってヒートマップを描画。
        """
        self.canvas.plot_solution(result.x, result.y, result.u)
        self.info_label.setText(f"Heat2D Viewer | run_id={result.run_id}")


# ============================================
# Control Panel（左側：パラメータ＋Run＋ログ）
# ============================================

class Heat2DControlPanel(QWidget):
    """
    Heat2D シミュレーションのパラメータ入力・実行とログ表示を担当するパネル。
    """

    simulationFinished = Signal(object)  # Heat2DResult を流す
    simulationFailed = Signal(str)       # エラーメッセージなどを流す（必要なら使用）

    def __init__(self, parent=None):
        super().__init__(parent)

        self.state = Heat2DState()

        main_layout = QVBoxLayout(self)

        # ---- フォーム ----
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

        main_layout.addLayout(form_layout)

        # ---- ボタン行 ----
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run 2D Heat Simulation")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        self.run_id_label = QLabel("run_id: (not run yet)")
        button_layout.addWidget(self.run_id_label)

        self.info_label = QLabel(self.state.info_message)
        button_layout.addWidget(self.info_label)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # ---- ログエリア ----
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_text)

    # ---------- ユーティリティ ----------

    def append_log(self, text: str) -> None:
        self.log_text.append(text)

    def _collect_config_dict(self) -> dict:
        """
        UI から設定値を収集し、backend にそのまま渡せる dict を返す。
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
        return cfg

    # ---------- Run ボタン ----------

    @Slot()
    def on_run_clicked(self) -> None:
        # Config を取得
        cfg_dict = self._collect_config_dict()

        # 状態リセット
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

            # 必要なら外部にも通知
            self.simulationFailed.emit(repr(e))
            return

        # 成功時
        self.state.is_running = False
        self.state.last_run_id = result.run_id
        self.state.info_message = "Done."

        self.run_id_label.setText(f"run_id: {result.run_id}")
        self.info_label.setText(self.state.info_message)

        if result.config_json_path is not None:
            self.append_log(f"Config JSON: {result.config_json_path}")
        if result.eval_json_path is not None:
            self.append_log(f"Eval JSON: {result.eval_json_path}")
        if result.summary_json_path is not None:
            self.append_log(f"Summary JSON: {result.summary_json_path}")

        self.run_button.setEnabled(True)

        # Viewer 側へ結果を通知
        self.simulationFinished.emit(result)


# ============================================
# Workspace タブ本体（左右分割）
# ============================================

class Heat2DTab(QWidget):
    """
    Heat2D 用ワークスペース。
    左に ControlPanel、右に Viewer を QSplitter で配置する。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal, self)

        self.control_panel = Heat2DControlPanel(splitter)
        self.viewer = Heat2DViewer(splitter)

        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.viewer)

        # 左パネルはやや狭く、右 Viewer を広く取る
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # シミュレーション完了時に Viewer に渡す
        self.control_panel.simulationFinished.connect(self.on_simulation_finished)

    @Slot(object)
    def on_simulation_finished(self, result_obj: object) -> None:
        # Signal は object なので Heat2DResult にキャストして使う
        if isinstance(result_obj, Heat2DResult):
            self.viewer.show_result(result_obj)
        else:
            # 想定外の型の場合は何もしない（もしくはログに出してもOK）
            pass
