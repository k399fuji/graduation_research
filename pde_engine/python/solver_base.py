# python/solver_base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np


class ReferenceSolver1D(ABC):
    """
    C++ ベースの 1D PDE 参照ソルバの共通インターフェース。

    すべての参照ソルバは solve() を実装して、
    (x, u(T, x)) を 1D numpy array で返す。
    """

    @abstractmethod
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class HeatSolverConfig:
    """
    1D 熱方程式 C++ ソルバの設定。
    今は最低限のパラメータだけだが、将来ここに追加していく。
    """

    L: float
    alpha: float
    T_final: float
    Nx_cpp: int
    dt_cpp: float
    ic_type: str = "gaussian"   # "gaussian" / "sine" など
    gaussian_k: float = 100.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeatSolverConfig":
        """
        pinn_heat1d.py の config_dict から必要な情報だけを抜き出す。
        """
        return cls(
            L=float(d["L"]),
            alpha=float(d["alpha"]),
            T_final=float(d["T_final"]),
            Nx_cpp=int(d["Nx_cpp"]),
            dt_cpp=float(d["dt_cpp"]),
        )
