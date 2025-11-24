# editor/workers.py
from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Any

from PySide6.QtCore import QThread, Signal

from backend.heat_pinn_backend import HeatPINNConfig, HeatPINNResult, run_heat_pinn
from backend.wave_pinn_backend import WavePINNConfig, WavePINNResult, run_wave_pinn


# ==============================
# Base worker
# ==============================

class BasePINNWorker(QThread):
    """
    Heat / Wave 共通の PINN 学習 Worker 基底クラス。

    - cfg      : dataclass などの設定オブジェクト
    - run_func : run_func(cfg, progress_callback=cb) という形の関数
    - start_msg / finish_msg : ログ用メッセージ
    """

    finished = Signal(object)       # Result (or None)
    message = Signal(str)           # ログメッセージ
    progress = Signal(int, float)   # epoch, loss_total

    def __init__(
        self,
        cfg: Any,
        run_func: Callable[..., Any],
        start_msg: str,
        finish_msg: str,
        parent=None,
    ):
        super().__init__(parent)
        self.cfg = cfg
        self.run_func = run_func
        self.start_msg = start_msg
        self.finish_msg = finish_msg

    def run(self):
        try:
            self.message.emit(self.start_msg)

            def cb(epoch, loss):
                self.progress.emit(int(epoch), float(loss))

            # backend 側の run_*_pinn を実行
            result = self.run_func(self.cfg, progress_callback=cb)

            self.message.emit(self.finish_msg)
            self.finished.emit(result)
        except Exception as e:
            self.message.emit(f"エラー発生: {e!r}")
            self.finished.emit(None)


# ==============================
# Heat / Wave 用の薄いラッパー
# ==============================

class HeatPINNWorker(BasePINNWorker):
    """
    既存コードとの互換のためのラッパー。
    """

    def __init__(self, cfg: HeatPINNConfig, parent=None):
        super().__init__(
            cfg=cfg,
            run_func=run_heat_pinn,
            start_msg="Heat PINN 実験を開始します...",
            finish_msg="Heat PINN 実験が完了しました。",
            parent=parent,
        )


class WavePINNWorker(BasePINNWorker):
    """
    既存コードとの互換のためのラッパー。
    """

    def __init__(self, cfg: WavePINNConfig, parent=None):
        super().__init__(
            cfg=cfg,
            run_func=run_wave_pinn,
            start_msg="Wave PINN 実験を開始します...",
            finish_msg="Wave PINN 実験が完了しました。",
            parent=parent,
        )
