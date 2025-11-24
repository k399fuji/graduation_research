# editor/viewmodels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ViewportState:
    """1D ビュー用の状態"""
    current_run_id: Optional[str] = None
    current_mode: str = "Solution"
    anim_step: int = 0


@dataclass
class TrainingState:
    """
    PINN 学習パネル用の ViewModel。
    UI とは切り離して「学習状態」だけを保持する。
    """
    is_running: bool = False          # 学習スレッドが動いているか
    total_epochs: int = 0            # 予定 epoch 数
    current_epoch: int = 0           # 今どこまで進んだか
    last_loss: float | None = None   # 直近の loss
    last_run_id: Optional[str] = None  # 最後に成功した run_id


__all__ = ["ViewportState", "TrainingState"]
