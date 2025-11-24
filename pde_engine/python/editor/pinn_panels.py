from __future__ import annotations

import sys
import os
import json
import numpy as np
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
from backend.heat2d_backend import Heat2DConfig, Heat2DResult, run_heat2d
try:
    import heat2d_cpp
except ImportError:
    heat2d_cpp = None
from backend import log_utils  # Logs / ファイルパス用
from reference_solvers import make_reference_solver
from editor.plugin_core import EditorPlugin


LOSS_LINEWIDTH = 0.5

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
        main_window.addDockWidget(Qt.LeftDockWidgetArea, dock)

        if hasattr(main_window, "view_menu"):
            main_window.view_menu.addAction(dock.toggleViewAction())

        main_window.pinn_dock = dock
        dock.hide()
        
        return dock
