from __future__ import annotations

import sys
import os
import json
from dataclasses import asdict

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QMessageBox,
    QListWidget,
    QProgressBar,
    QDockWidget,
    QComboBox,
    QSlider,
    QTabWidget,
)

# --- プロジェクトの python/ を import パスに追加 ---
CURRENT_DIR = os.path.dirname(__file__)
PYTHON_DIR = os.path.dirname(CURRENT_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.append(PYTHON_DIR)

from backend.heat_pinn_backend import HeatPINNConfig, HeatPINNResult, run_heat_pinn
from backend.wave_pinn_backend import WavePINNConfig, WavePINNResult, run_wave_pinn
from backend import log_utils  # Logs / ファイルパス用
from editor.plugin_core import EditorPlugin, PluginManager

LOSS_LINEWIDTH = 0.5


# ============================
#  共通: Matplotlib キャンバス
# ============================

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
            
        ax.tick_params(axis='both', colors="white")
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        for spine in ax.spines.values():
            spine.set_color("white")
            
        ax.title.set_color("white")

# ============================
#  Viewport（中央の表示領域）
# ============================

class ViewportWidget(QWidget):
    """
    中央ビュー。
    指定された run_id の eval.json / CSV を読み込み、モードに応じて
      - Solution: u_pinn と u_cpp
      - Error   : |u_pinn - u_cpp_interp|
      - Loss    : loss_total（CSV）
      - Animation: 時間発展アニメーション（PINNで再計算）
    をプロットする。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_run_id: str | None = None
        self._current_anim_step: int = 0
        self._anim_state: dict | None = None

        layout = QVBoxLayout(self)

        # --- 上部: モード切り替え ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("View:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Solution", "Error", "Loss", "Animation"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # --- 中央: Matplotlib キャンバス ---
        self.canvas = MatplotlibCanvas(self, width=6, height=4, dpi=100)
        layout.addWidget(self.canvas, 1)

        # --- 下部: 時間スライダー (Animation 用)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.setValue(100)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.time_slider.setVisible(False)  # 初期状態では非表示
        layout.addWidget(self.time_slider)

        # --- 下部: 情報ラベル ---
        self.info_label = QLabel("No run yet")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

    # ====== 共通ヘルパー ======

    def _load_eval(self, run_id: str) -> dict | None:
        """
        eval.json を読み込んで dict を返す。
        読み込み失敗時は None。
        """
        eval_path = log_utils.path_eval(run_id)
        if not eval_path.exists():
            return None
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ====== Animation 用セットアップ ======

    def _setup_animation_state(self, run_id: str, ax):
        """
        run_id に対応する config.json / model.pt を読み込み、
        アニメーションで使う状態(self._anim_state)を準備する。
        """
        import torch
        import numpy as np  # 将来拡張用（今は未使用だが保持）

        config_path = log_utils.path_config(run_id)
        model_path = log_utils.path_model(run_id)

        if not config_path.exists() or not model_path.exists():
            ax.text(0.5, 0.5, "No config/model for this run",
                    ha="center", va="center")
            self.info_label.setText(f"{run_id}: no model checkpoint")
            self._anim_state = None
            return

        # config 読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # summary から backend 種類を取得（なければ wave/heat を推定）
        summary = log_utils.load_summary(run_id)
        backend = None
        if summary is not None:
            backend = summary.get("run", {}).get("backend", None)

        # backend から使用する PINN モジュールを決定
        if backend == "pinn_wave1d":
            import pinn_wave1d as pinn_module
        else:
            # デフォルトは heat
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

        # アニメーション用のメタ情報
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

        # スライダー設定と初期フレーム描画（T_final 側にしておく）
        self.time_slider.blockSignals(True)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(N_frames)
        self.time_slider.setValue(N_frames)
        self.time_slider.blockSignals(False)

        self._update_animation_frame(N_frames, ax)

    # ===== 外部 API =====

    def show_result(self, run_id: str):
        """PINN 実験完了時に呼ばれる。内部状態を更新して再描画。"""
        self._current_run_id = run_id
        self._update_plot()

    # ===== モード変更スロット =====

    @Slot()
    def on_mode_changed(self):
        """コンボボックス変更時に再描画。"""
        if self._current_run_id is not None:
            self._update_plot()

    # ====== モード別描画 ======

    def _plot_solution(self, ax, run_id: str, ev: dict | None):
        import numpy as np

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

    def _plot_error(self, ax, run_id: str, ev: dict | None):
        import numpy as np

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
        """
        CSV ログから Loss 曲線を描画する。
        """
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
        """
        Animation モードの入り口。
        state が None の場合は eval.json ベースの静止図にフォールバック。
        """
        self._setup_animation_state(run_id, ax)
        # _setup_animation_state 内で描画まで行うのでここでは何もしない

    # ====== メインの描画更新 ======

    def _update_plot(self):
        """現在の run_id とモードにもとづいてグラフを更新する。"""
        ax = self.canvas.axes
        ax.clear()

        run_id = self._current_run_id
        if not run_id:
            ax.text(0.5, 0.5, "No run selected", ha="center", va="center")
            self.info_label.setText("No run yet")
            self.canvas.draw()
            return

        mode = self.mode_combo.currentText()
        self.time_slider.setVisible(mode == "Animation")

        ev = None
        if mode in ("Solution", "Error"):
            ev = self._load_eval(run_id)
            if ev is None:
                # 従来のメッセージを踏襲しつつラベルだけ整える
                ax.text(0.5, 0.5, "No eval.json for this run",
                        ha="center", va="center")

        if mode == "Solution":
            self._plot_solution(ax, run_id, ev)
        elif mode == "Error":
            self._plot_error(ax, run_id, ev)
        elif mode == "Loss":
            self._plot_loss(ax, run_id)
        elif mode == "Animation":
            self._plot_animation_or_fallback(ax, run_id)
            self.canvas.draw()
            return
        else:
            ax.text(0.5, 0.5, f"Unknown mode: {mode}",
                    ha="center", va="center")
            self.info_label.setText(f"Unknown mode: {mode}")
        
        self.canvas.apply_dark_style()
        self.canvas.draw()

    def _update_animation_frame(self, step: int, ax=None):
        """
        Animation モード用の描画処理。

        - self._anim_state がある場合:
            保存してある PINN モデルを使って、スライダー位置に応じた時刻 t で
            u_pinn(x, t) を計算して描画する。
        - self._anim_state が None の場合:
            eval.json の最終時刻データだけを描画する
            （スライダーを動かしても同じ絵だが、最低限エラーは出ない）。
        """
        import numpy as np

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

        # ================================
        # 1) PINN モデルを使ったアニメーション
        # ================================
        if state is not None:
            try:
                import torch
            except ImportError:
                state = None
            else:
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

        # ==========================================
        # 2) フォールバック: eval.json の最終時刻を描画
        # ==========================================
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


# ============================
#  計算用 Worker スレッド
# ============================

class HeatPINNWorker(QThread):
    finished = Signal(object)        # HeatPINNResult
    message = Signal(str)            # ログメッセージ
    progress = Signal(int, float)    # epoch, loss_total

    def __init__(self, cfg: HeatPINNConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            self.message.emit("Heat PINN 実験を開始します...")

            def cb(epoch, loss):
                self.progress.emit(int(epoch), float(loss))

            result = run_heat_pinn(self.cfg, progress_callback=cb)

            self.message.emit("Heat PINN 実験が完了しました。")
            self.finished.emit(result)
        except Exception as e:
            self.message.emit(f"エラー発生: {e!r}")
            self.finished.emit(None)


class WavePINNWorker(QThread):
    finished = Signal(object)        # WavePINNResult
    message = Signal(str)
    progress = Signal(int, float)

    def __init__(self, cfg: WavePINNConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            self.message.emit("Wave PINN 実験を開始します...")

            def cb(epoch, loss):
                self.progress.emit(int(epoch), float(loss))

            result = run_wave_pinn(self.cfg, progress_callback=cb)

            self.message.emit("Wave PINN 実験が完了しました。")
            self.finished.emit(result)
        except Exception as e:
            self.message.emit(f"エラー発生: {e!r}")
            self.finished.emit(None)


# ============================
#  Heat PINN パネル
# ============================

class HeatPINNPage(QWidget):
    runFinished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        # 上：設定フォーム
        form_layout = QFormLayout()
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.1, 10.0)
        self.spin_L.setValue(1.0)
        self.spin_L.setSingleStep(0.1)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setDecimals(5)
        self.spin_alpha.setRange(1e-5, 1.0)
        self.spin_alpha.setValue(0.01)
        self.spin_alpha.setSingleStep(0.001)

        self.spin_T = QDoubleSpinBox()
        self.spin_T.setRange(0.01, 10.0)
        self.spin_T.setValue(0.4)
        self.spin_T.setSingleStep(0.1)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(100, 100000)
        self.spin_epochs.setValue(5000)
        self.spin_epochs.setSingleStep(100)

        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(8, 512)
        self.spin_hidden.setValue(64)
        self.spin_hidden.setSingleStep(8)

        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(2, 10)
        self.spin_layers.setValue(4)

        self.spin_Nr = QSpinBox()
        self.spin_Nr.setRange(100, 50000)
        self.spin_Nr.setValue(1000)
        self.spin_Nr.setSingleStep(100)

        self.spin_Nic = QSpinBox()
        self.spin_Nic.setRange(10, 10000)
        self.spin_Nic.setValue(200)
        self.spin_Nic.setSingleStep(10)

        self.spin_Nbc = QSpinBox()
        self.spin_Nbc.setRange(10, 10000)
        self.spin_Nbc.setValue(200)
        self.spin_Nbc.setSingleStep(10)

        self.combo_ic_type = QComboBox()
        self.combo_ic_type.addItems(["gaussian", "sine", "twopeaks"])

        self.spin_gaussian_k = QDoubleSpinBox()
        self.spin_gaussian_k.setRange(1.0, 500.0)
        self.spin_gaussian_k.setValue(100.0)
        self.spin_gaussian_k.setSingleStep(10.0)

        form_layout.addRow("IC type", self.combo_ic_type)
        form_layout.addRow("Gaussian k", self.spin_gaussian_k)

        self.edit_tag = QLineEdit("qt")

        form_layout.addRow("L (domain length)", self.spin_L)
        form_layout.addRow("alpha (diffusivity)", self.spin_alpha)
        form_layout.addRow("T_final", self.spin_T)
        form_layout.addRow("epochs", self.spin_epochs)
        form_layout.addRow("hidden_dim", self.spin_hidden)
        form_layout.addRow("num_layers", self.spin_layers)
        form_layout.addRow("N_r (residual points)", self.spin_Nr)
        form_layout.addRow("N_ic (IC points)", self.spin_Nic)
        form_layout.addRow("N_bc (BC points)", self.spin_Nbc)

        # 損失重み
        self.spin_w_pde = QDoubleSpinBox()
        self.spin_w_pde.setRange(0.0, 100.0)
        self.spin_w_pde.setSingleStep(0.1)
        self.spin_w_pde.setValue(1.0)

        self.spin_w_ic = QDoubleSpinBox()
        self.spin_w_ic.setRange(0.0, 100.0)
        self.spin_w_ic.setSingleStep(0.1)
        self.spin_w_ic.setValue(1.0)

        self.spin_w_bc = QDoubleSpinBox()
        self.spin_w_bc.setRange(0.0, 100.0)
        self.spin_w_bc.setSingleStep(0.1)
        self.spin_w_bc.setValue(1.0)

        form_layout.addRow("w_pde (PDE weight)", self.spin_w_pde)
        form_layout.addRow("w_ic (IC weight)", self.spin_w_ic)
        form_layout.addRow("w_bc (BC weight)", self.spin_w_bc)

        form_layout.addRow("tag", self.edit_tag)

        main_layout.addLayout(form_layout)

        # 実行ボタン＋run_id 表示
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Heat PINN")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        self.run_id_label = QLabel("run_id: (not run yet)")
        button_layout.addWidget(self.run_id_label)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # 進捗バー＆ラベル
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # 初期値
        self.progress_bar.setValue(0)

        self.progress_label = QLabel("Epoch: 0 / 0")
        progress_layout.addWidget(QLabel("Training progress:"))
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.progress_label)

        main_layout.addLayout(progress_layout)

        # 下：ログ表示エリア
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_text)

        self._worker: HeatPINNWorker | None = None

    def append_log(self, text: str):
        self.log_text.append(text)

    def get_config(self) -> HeatPINNConfig:
        return HeatPINNConfig(
            L=float(self.spin_L.value()),
            alpha=float(self.spin_alpha.value()),
            T_final=float(self.spin_T.value()),
            epochs=int(self.spin_epochs.value()),
            hidden_dim=int(self.spin_hidden.value()),
            num_layers=int(self.spin_layers.value()),
            N_r=int(self.spin_Nr.value()),
            N_ic=int(self.spin_Nic.value()),
            N_bc=int(self.spin_Nbc.value()),
            w_pde=float(self.spin_w_pde.value()),
            w_ic=float(self.spin_w_ic.value()),
            w_bc=float(self.spin_w_bc.value()),
            ic_type=self.combo_ic_type.currentText(),
            gaussian_k=float(self.spin_gaussian_k.value()),
            tag=self.edit_tag.text().strip() or "qt",
        )

    @Slot()
    def on_run_clicked(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "実行中", "すでに実行中です。完了をお待ちください。")
            return

        cfg = self.get_config()
        self.append_log(f"Config: {asdict(cfg)}")
        self.append_log("Heat PINN 実験を開始します。")

        # ProgressBar 初期化
        self.progress_bar.setMaximum(cfg.epochs)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Epoch: 0 / {cfg.epochs}")

        self.run_button.setEnabled(False)

        self._worker = HeatPINNWorker(cfg, self)
        self._worker.message.connect(self.append_log)
        self._worker.progress.connect(self.on_worker_progress)
        self._worker.finished.connect(self.on_worker_finished)
        self._worker.start()

    @Slot(int, float)
    def on_worker_progress(self, epoch: int, loss: float):
        max_epoch = self.progress_bar.maximum()
        if max_epoch <= 0:
            return
        if epoch > max_epoch:
            epoch = max_epoch

        self.progress_bar.setValue(epoch)
        self.progress_label.setText(
            f"Epoch: {epoch} / {max_epoch}   loss={loss:.3e}"
        )

    @Slot(object)
    def on_worker_finished(self, result: HeatPINNResult | None):
        self.run_button.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())

        if result is None:
            self.append_log("エラーにより実験が終了しました。")
            QMessageBox.critical(
                self,
                "エラー",
                "Heat PINN 実験中にエラーが発生しました。ターミナルのログも確認してください。",
            )
            return

        self.run_id_label.setText(f"run_id: {result.run_id}")
        self.append_log(f"実験完了。run_id = {result.run_id}")
        if result.log_csv_path is not None:
            self.append_log(f"Log CSV: {result.log_csv_path}")
        if result.config_json_path is not None:
            self.append_log(f"Config JSON: {result.config_json_path}")
        if result.eval_json_path is not None:
            self.append_log(f"Eval JSON: {result.eval_json_path or '(not created yet)'}")

        self.runFinished.emit(result.run_id)
        QMessageBox.information(self, "完了", "Heat PINN 実験が完了しました。")


class WavePINNPage(QWidget):
    runFinished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        # PDE params
        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.1, 10.0)
        self.spin_L.setValue(1.0)
        self.spin_L.setSingleStep(0.1)

        self.spin_c = QDoubleSpinBox()
        self.spin_c.setRange(0.1, 10.0)
        self.spin_c.setValue(1.0)
        self.spin_c.setSingleStep(0.1)

        self.spin_T = QDoubleSpinBox()
        self.spin_T.setRange(0.01, 10.0)
        self.spin_T.setValue(1.0)
        self.spin_T.setSingleStep(0.1)

        # PINN params
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(100, 100000)
        self.spin_epochs.setValue(5000)
        self.spin_epochs.setSingleStep(100)

        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(8, 512)
        self.spin_hidden.setValue(64)
        self.spin_hidden.setSingleStep(8)

        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(2, 10)
        self.spin_layers.setValue(4)

        # sampling
        self.spin_Nr = QSpinBox()
        self.spin_Nr.setRange(100, 50000)
        self.spin_Nr.setValue(2000)
        self.spin_Nr.setSingleStep(100)

        self.spin_Nic_u = QSpinBox()
        self.spin_Nic_u.setRange(10, 10000)
        self.spin_Nic_u.setValue(200)
        self.spin_Nic_u.setSingleStep(10)

        self.spin_Nic_ut = QSpinBox()
        self.spin_Nic_ut.setRange(10, 10000)
        self.spin_Nic_ut.setValue(200)
        self.spin_Nic_ut.setSingleStep(10)

        self.spin_Nbc = QSpinBox()
        self.spin_Nbc.setRange(10, 10000)
        self.spin_Nbc.setValue(200)
        self.spin_Nbc.setSingleStep(10)

        # IC
        self.combo_ic_type = QComboBox()
        self.combo_ic_type.addItems(["gaussian", "sine", "twopeaks"])

        self.spin_gaussian_k = QDoubleSpinBox()
        self.spin_gaussian_k.setRange(1.0, 500.0)
        self.spin_gaussian_k.setValue(100.0)
        self.spin_gaussian_k.setSingleStep(10.0)

        self.edit_tag = QLineEdit("qt")

        # loss weights
        self.spin_w_pde = QDoubleSpinBox()
        self.spin_w_pde.setRange(0.0, 100.0)
        self.spin_w_pde.setSingleStep(0.1)
        self.spin_w_pde.setValue(1.0)

        self.spin_w_ic_u = QDoubleSpinBox()
        self.spin_w_ic_u.setRange(0.0, 100.0)
        self.spin_w_ic_u.setSingleStep(0.1)
        self.spin_w_ic_u.setValue(1.0)

        self.spin_w_ic_ut = QDoubleSpinBox()
        self.spin_w_ic_ut.setRange(0.0, 100.0)
        self.spin_w_ic_ut.setSingleStep(0.1)
        self.spin_w_ic_ut.setValue(1.0)

        self.spin_w_bc = QDoubleSpinBox()
        self.spin_w_bc.setRange(0.0, 100.0)
        self.spin_w_bc.setSingleStep(0.1)
        self.spin_w_bc.setValue(1.0)

        # ---- form layout ----
        form_layout.addRow("IC type", self.combo_ic_type)
        form_layout.addRow("Gaussian k", self.spin_gaussian_k)

        form_layout.addRow("L (domain length)", self.spin_L)
        form_layout.addRow("c (wave speed)", self.spin_c)
        form_layout.addRow("T_final", self.spin_T)

        form_layout.addRow("epochs", self.spin_epochs)
        form_layout.addRow("hidden_dim", self.spin_hidden)
        form_layout.addRow("num_layers", self.spin_layers)

        form_layout.addRow("N_r (residual points)", self.spin_Nr)
        form_layout.addRow("N_ic_u (IC u points)", self.spin_Nic_u)
        form_layout.addRow("N_ic_ut (IC ut points)", self.spin_Nic_ut)
        form_layout.addRow("N_bc (BC points)", self.spin_Nbc)

        form_layout.addRow("w_pde", self.spin_w_pde)
        form_layout.addRow("w_ic_u", self.spin_w_ic_u)
        form_layout.addRow("w_ic_ut", self.spin_w_ic_ut)
        form_layout.addRow("w_bc", self.spin_w_bc)

        form_layout.addRow("tag", self.edit_tag)

        main_layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Wave PINN")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        self.run_id_label = QLabel("run_id: (not run yet)")
        button_layout.addWidget(self.run_id_label)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)

        self.progress_label = QLabel("Epoch: 0 / 0")
        progress_layout.addWidget(QLabel("Training progress:"))
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_text)

        self._worker: WavePINNWorker | None = None

    def append_log(self, text: str):
        self.log_text.append(text)

    def get_config(self) -> WavePINNConfig:
        return WavePINNConfig(
            L=float(self.spin_L.value()),
            c=float(self.spin_c.value()),
            T_final=float(self.spin_T.value()),
            epochs=int(self.spin_epochs.value()),
            hidden_dim=int(self.spin_hidden.value()),
            num_layers=int(self.spin_layers.value()),
            N_r=int(self.spin_Nr.value()),
            N_ic_u=int(self.spin_Nic_u.value()),
            N_ic_ut=int(self.spin_Nic_ut.value()),
            N_bc=int(self.spin_Nbc.value()),
            w_pde=float(self.spin_w_pde.value()),
            w_ic_u=float(self.spin_w_ic_u.value()),
            w_ic_ut=float(self.spin_w_ic_ut.value()),
            w_bc=float(self.spin_w_bc.value()),
            ic_type=self.combo_ic_type.currentText(),
            gaussian_k=float(self.spin_gaussian_k.value()),
            tag=self.edit_tag.text().strip() or "qt",
        )

    @Slot()
    def on_run_clicked(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "実行中", "すでに実行中です。完了をお待ちください。")
            return

        cfg = self.get_config()
        self.append_log(str(cfg))
        self.append_log("Wave PINN 実験を開始します。")

        self.progress_bar.setMaximum(cfg.epochs)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Epoch: 0 / {cfg.epochs}")

        self.run_button.setEnabled(False)

        self._worker = WavePINNWorker(cfg, self)
        self._worker.message.connect(self.append_log)
        self._worker.progress.connect(self.on_worker_progress)
        self._worker.finished.connect(self.on_worker_finished)
        self._worker.start()

    @Slot(int, float)
    def on_worker_progress(self, epoch: int, loss: float):
        max_epoch = self.progress_bar.maximum()
        if max_epoch <= 0:
            return
        if epoch > max_epoch:
            epoch = max_epoch

        self.progress_bar.setValue(epoch)
        self.progress_label.setText(
            f"Epoch: {epoch} / {max_epoch}   loss={loss:.3e}"
        )

    @Slot(object)
    def on_worker_finished(self, result: WavePINNResult | None):
        self.run_button.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())

        if result is None:
            self.append_log("エラーにより実験が終了しました。")
            QMessageBox.critical(
                self,
                "エラー",
                "Wave PINN 実験中にエラーが発生しました。ターミナルのログも確認してください。",
            )
            return

        self.run_id_label.setText(f"run_id: {result.run_id}")
        self.append_log(f"実験完了。run_id = {result.run_id}")
        if result.log_csv_path is not None:
            self.append_log(f"Log CSV: {result.log_csv_path}")
        if result.config_json_path is not None:
            self.append_log(f"Config JSON: {result.config_json_path}")
        if result.eval_json_path is not None:
            self.append_log(f"Eval JSON: {result.eval_json_path or '(not created yet)'}")

        self.runFinished.emit(result.run_id)
        QMessageBox.information(self, "完了", "Wave PINN 実験が完了しました。")


# ============================
#  Logs パネル
# ============================

class LogsPage(QWidget):
    runSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QHBoxLayout(self)

        # 左：run_id の一覧
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Runs"))

        self.run_list = QListWidget()
        self.run_list.currentItemChanged.connect(self.on_run_selected)
        self.run_list.setMaximumWidth(185)
        left_layout.addWidget(self.run_list)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_runs)
        left_layout.addWidget(self.refresh_button)

        # 右：config / eval / loss グラフ
        right_layout = QVBoxLayout()

        right_layout.addWidget(QLabel("Config"))
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        right_layout.addWidget(self.config_text, 2)

        right_layout.addWidget(QLabel("Eval"))
        self.eval_label = QLabel("(no eval)")
        right_layout.addWidget(self.eval_label)

        right_layout.addWidget(QLabel("Loss curves"))
        self.canvas = MatplotlibCanvas(self, width=5, height=3, dpi=100)

        right_layout.addWidget(self.canvas, 4)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        # 初回ロード
        self.refresh_runs()

    def refresh_runs(self):
        self.run_list.clear()
        self._runs = log_utils.list_runs()
        if not self._runs:
            self.run_list.addItem("(no runs)")
            self.run_list.setEnabled(False)
            return

        self.run_list.setEnabled(True)
        for info in self._runs:
            self.run_list.addItem(info.run_id)

    @Slot("QListWidgetItem*", "QListWidgetItem*")
    def on_run_selected(self, current, previous):
        if current is None:
            return
        run_id = current.text()
        if run_id == "(no runs)":
            return

        # ---- Summary or Config を表示 ----
        summary = log_utils.load_summary(run_id)
        if summary is not None:
            self.config_text.setPlainText(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
        else:
            cfg = log_utils.load_config(run_id)
            if cfg is None:
                self.config_text.setPlainText("(no config.json)")
            else:
                self.config_text.setPlainText(
                    json.dumps(cfg, indent=2, ensure_ascii=False)
                )

        # ---- Eval を表示 ----
        ev = log_utils.load_eval(run_id)
        if ev is None:
            if summary is not None:
                metrics = summary.get("results", {}).get("metrics", {})
                l2 = metrics.get("L2_error", None)
                linf = metrics.get("Linf_error", None)
                if l2 is not None and linf is not None:
                    self.eval_label.setText(f"L2 = {l2:.3e},  L∞ = {linf:.3e}")
                else:
                    self.eval_label.setText("No eval.json")
            else:
                self.eval_label.setText("No eval.json")
        else:
            l2 = ev.get("L2_error", None)
            linf = ev.get("Linf_error", None)
            if l2 is not None and linf is not None:
                self.eval_label.setText(f"L2 = {l2:.3e},  L∞ = {linf:.3e}")
            else:
                self.eval_label.setText(json.dumps(ev, ensure_ascii=False))

        # ---- Loss グラフ描画 ----
        rows = log_utils.load_loss_csv(run_id)
        ax = self.canvas.axes
        ax.clear()

        if not rows:
            ax.text(0.5, 0.5, "No CSV log", ha="center", va="center")
        else:
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

            line_width = getattr(log_utils, "LINE_WIDTH", 1.5)

            if loss_total is not None:
                ax.plot(epochs, loss_total, label="loss_total",
                        linewidth=line_width)
            if loss_pde is not None:
                ax.plot(epochs, loss_pde, label="loss_pde",
                        linewidth=line_width)
            if loss_ic is not None:
                ax.plot(epochs, loss_ic, label="loss_ic",
                        linewidth=line_width)
            if loss_bc is not None:
                ax.plot(epochs, loss_bc, label="loss_bc",
                        linewidth=line_width)

            ax.set_xlabel("epoch", color="white")
            ax.set_ylabel("loss", color="white")
            ax.set_yscale("log")
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

        self.canvas.fig.subplots_adjust(bottom=0.18)
        self.canvas.apply_dark_style()
        self.canvas.draw()
        self.runSelected.emit(run_id)


# ============================
#  プラグイン実装
# ============================

class HeatPINNPlugin(EditorPlugin):
    plugin_id = "heat_pinn"
    display_name = "Heat PINN"

    def create_dock(self, main_window: QMainWindow) -> QDockWidget:
        widget = HeatPINNPage(main_window)

        if hasattr(main_window, "on_pinn_run_finished"):
            widget.runFinished.connect(main_window.on_pinn_run_finished)

        dock = QDockWidget(self.display_name, main_window)
        dock.setWidget(widget)
        dock.setObjectName(self.plugin_id)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        main_window.addDockWidget(Qt.LeftDockWidgetArea, dock)
        return dock


class WavePINNPlugin(EditorPlugin):
    plugin_id = "wave_pinn"
    display_name = "Wave PINN"

    def create_dock(self, main_window: QMainWindow) -> QDockWidget:
        widget = WavePINNPage(main_window)
        if hasattr(main_window, "on_pinn_run_finished"):
            widget.runFinished.connect(main_window.on_pinn_run_finished)

        dock = QDockWidget(self.display_name, main_window)
        dock.setWidget(widget)
        dock.setObjectName(self.plugin_id)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        main_window.addDockWidget(Qt.LeftDockWidgetArea, dock)
        return dock


class LogsPlugin(EditorPlugin):
    plugin_id = "logs"
    display_name = "Logs"

    def create_dock(self, main_window: QMainWindow) -> QDockWidget:
        widget = LogsPage(main_window)

        if hasattr(main_window, "on_log_run_selected"):
            widget.runSelected.connect(main_window.on_log_run_selected)

        dock = QDockWidget(self.display_name, main_window)
        dock.setWidget(widget)
        dock.setObjectName(self.plugin_id)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.hide()
        if hasattr(main_window, "view_menu"):
            main_window.view_menu.addAction(dock.toggleViewAction())
        return dock

class PINNTabbedPlugin(EditorPlugin):
    """
    Heat / Wave の両方をタブで切り替えられる 1 つのドック。
    左側に常駐させる。
    """
    plugin_id = "pinn_tabbed"
    display_name = "PINN"

    def create_dock(self, main_window: QMainWindow) -> QDockWidget:
        # タブウィジェット本体
        tab_widget = QTabWidget(main_window)

        # Heat / Wave の各ページを作成
        heat_page = HeatPINNPage(tab_widget)
        wave_page = WavePINNPage(tab_widget)

        # タブに追加（左タブがデフォルトで Heat）
        tab_widget.addTab(heat_page, "Heat")
        tab_widget.addTab(wave_page, "Wave")

        # メインウィンドウ側のスロットに runFinished を接続
        if hasattr(main_window, "on_pinn_run_finished"):
            heat_page.runFinished.connect(main_window.on_pinn_run_finished)
            wave_page.runFinished.connect(main_window.on_pinn_run_finished)

        # ドックとして包む
        dock = QDockWidget(self.display_name, main_window)
        dock.setWidget(tab_widget)
        dock.setObjectName(self.plugin_id)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # 左側に配置
        main_window.addDockWidget(Qt.LeftDockWidgetArea, dock)
        if hasattr(main_window, "view_menu"):
            main_window.view_menu.addAction(dock.toggleViewAction())
        return dock


def apply_dark_theme(app: QApplication) -> None:
    """
    Qt 全体をダークテーマにする。
    """
    app.setStyle("Fusion")

    palette = QPalette()

    # ウィンドウ全体
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)

    # ベース（テキストエリアや入力欄などの背景）
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))

    # テキスト
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)

    # ボタン
    palette.setColor(QPalette.Button, QColor(53, 53, 53))

    # ハイライト（選択色）
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    # 他（リンク色など）
    palette.setColor(QPalette.Link, QColor(42, 130, 218))

    app.setPalette(palette)


# ============================
#  メインウィンドウ
# ============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDE Engine & PINN Editor (Qt)")

        # 中央ビュー（シミュレーション表示）
        self.viewport = ViewportWidget(self)
        self.setCentralWidget(self.viewport)

        menu_bar = self.menuBar()
        self.view_menu = menu_bar.addMenu("&View")

        # プラグインマネージャ
        self.plugin_manager = PluginManager(self)
        self._register_builtin_plugins()

        self.resize(1200, 800)

    def _register_builtin_plugins(self):
        self.plugin_manager.register_plugin(PINNTabbedPlugin)
        self.plugin_manager.register_plugin(LogsPlugin)

    @Slot(str)
    def on_pinn_run_finished(self, run_id: str):
        self.viewport.show_result(run_id)

    @Slot(str)
    def on_log_run_selected(self, run_id: str):
        self.viewport.show_result(run_id)


# ============================
#  エントリポイント
# ============================

def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
