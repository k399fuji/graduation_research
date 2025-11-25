# editor/viewmodels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ===== 1D Viewport 用 =====

@dataclass
class ViewportState:
    """1D 中央ビューの状態"""
    current_run_id: Optional[str] = None
    current_mode: str = "Solution"
    info_message: str = "No run yet"
    current_anim_step: int = 0


# ===== PINN Training 用 =====

@dataclass
class TrainingState:
    """Heat/Wave PINN 共通の訓練状態"""
    is_running: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    last_loss: Optional[float] = None
    last_run_id: Optional[str] = None
    message: str = ""


# ===== Heat2D 用 =====

@dataclass
class Heat2DState:
    """2D Heat タブの状態"""
    is_running: bool = False
    last_run_id: Optional[str] = None
    last_error: Optional[str] = None
    info_message: str = "Ready."
