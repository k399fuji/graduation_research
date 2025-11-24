# editor/pinn_base.py
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, TypeVar, Generic

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QProgressBar,
    QTextEdit,
    QMessageBox,
)

from editor.viewmodels import TrainingState
from editor.workers import BasePINNWorker


TConfig = TypeVar("TConfig")


class BasePINNPage(QWidget, Generic[TConfig]):
    """
    Heat / Wave 共通の PINN 設定パネルのベースクラス。

    - UI（フォーム・ボタン・プログレスバー・ログ）
    - 学習状態（TrainingState）
    - Worker とのつなぎ込み

    をまとめて面倒を見る。
    派生クラス側は

      * experiment_name
      * build_form(...)
      * get_config()
      * create_worker(cfg)

    だけ実装すればよい。
    """

    runFinished = Signal(str)  # 成功時に run_id を流す

    # 派生クラスで上書きする識別名
    experiment_name: str = "PINN"

    def __init__(self, parent=None):
        super().__init__(parent)

        # ==== ViewModel ====
        self.state = TrainingState()

        # ==== UI 構築 ====
        main_layout = QVBoxLayout(self)

        # フォーム部分（派生クラスに作らせる）
        form_layout = QFormLayout()
        self.build_form(form_layout)
        main_layout.addLayout(form_layout)

        # ボタン行
        button_layout = QHBoxLayout()
        self.run_button = QPushButton(f"Run {self.experiment_name} PINN")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        self.run_id_label = QLabel("run_id: (not run yet)")
        button_layout.addWidget(self.run_id_label)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # 進捗表示
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Training progress:"))

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, 1)

        self.progress_label = QLabel("Epoch: 0 / 0")
        progress_layout.addWidget(self.progress_label)

        main_layout.addLayout(progress_layout)

        # ログエリア
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_text)

        # Worker
        self._worker: Optional[BasePINNWorker] = None

    # ======= 抽象メソッド（派生クラス実装） =======

    def build_form(self, form_layout: QFormLayout) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_config(self) -> TConfig:  # pragma: no cover - abstract
        raise NotImplementedError

    def create_worker(self, cfg: TConfig) -> BasePINNWorker:  # pragma: no cover - abstract
        raise NotImplementedError

    # ======= 共通ヘルパ =======

    def append_log(self, text: str) -> None:
        self.log_text.append(text)

    # ======= ボタンクリック → 学習開始 =======

    @Slot()
    def on_run_clicked(self):
        # 既に学習中であれば弾く
        if (
            self.state.is_running
            and self._worker is not None
            and self._worker.isRunning()
        ):
            QMessageBox.warning(self, "実行中", "すでに実行中です。完了をお待ちください。")
            return

        cfg = self.get_config()

        # epochs を安全に取得（無ければ 0）
        try:
            epochs = int(getattr(cfg, "epochs"))
        except Exception:
            epochs = 0

        # --- ViewModel 初期化 ---
        self.state.is_running = True
        self.state.total_epochs = epochs
        self.state.current_epoch = 0
        self.state.last_loss = None
        self.state.last_run_id = None

        # --- UI 初期化 ---
        self.progress_bar.setMaximum(epochs if epochs > 0 else 0)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Epoch: 0 / {epochs}")
        self.run_button.setEnabled(False)

        # config をログ出力（dataclass なら asdict、それ以外は repr）
        try:
            cfg_dict = asdict(cfg)
        except TypeError:
            cfg_dict = repr(cfg)
        self.append_log(f"Config: {cfg_dict}")
        self.append_log(f"{self.experiment_name} PINN 実験を開始します。")

        # --- Worker 起動 ---
        self._worker = self.create_worker(cfg)
        self._worker.message.connect(self.append_log)
        self._worker.progress.connect(self.on_worker_progress)
        self._worker.finished.connect(self.on_worker_finished)
        self._worker.start()

    # ======= Worker → 進捗通知 =======

    @Slot(int, float)
    def on_worker_progress(self, epoch: int, loss: float):
        # ViewModel 更新
        if self.state.total_epochs > 0 and epoch > self.state.total_epochs:
            epoch = self.state.total_epochs

        self.state.current_epoch = epoch
        self.state.last_loss = loss

        # UI 更新
        if self.state.total_epochs > 0:
            self.progress_bar.setMaximum(self.state.total_epochs)
        self.progress_bar.setValue(epoch)

        max_ep = self.state.total_epochs
        self.progress_label.setText(
            f"Epoch: {epoch} / {max_ep}   loss={loss:.3e}"
        )

    # ======= Worker 終了時 =======

    @Slot(object)
    def on_worker_finished(self, result):
        # ViewModel 更新
        self.state.is_running = False

        # UI の基本更新
        self.run_button.setEnabled(True)
        if self.state.total_epochs > 0:
            self.progress_bar.setValue(self.state.total_epochs)

        if result is None:
            self.append_log("エラーにより実験が終了しました。")
            QMessageBox.critical(
                self,
                "エラー",
                f"{self.experiment_name} PINN 実験中にエラーが発生しました。"
                "ターミナルのログも確認してください。",
            )
            return

        # run_id 反映
        run_id = getattr(result, "run_id", None)
        if run_id:
            self.state.last_run_id = run_id
            self.run_id_label.setText(f"run_id: {run_id}")
            self.append_log(f"実験完了。run_id = {run_id}")
        else:
            self.append_log("実験完了。run_id は取得できませんでした。")

        # 追加のパスをログに出す（あれば）
        for attr, label in [
            ("log_csv_path", "Log CSV"),
            ("config_json_path", "Config JSON"),
            ("eval_json_path", "Eval JSON"),
        ]:
            path = getattr(result, attr, None)
            if path is not None:
                self.append_log(f"{label}: {path}")

        # ViewModel に run_id が入っていればシグナルも流す
        if self.state.last_run_id:
            self.runFinished.emit(self.state.last_run_id)

        QMessageBox.information(
            self,
            "完了",
            f"{self.experiment_name} PINN 実験が完了しました。",
        )
