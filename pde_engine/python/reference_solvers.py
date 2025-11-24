from __future__ import annotations
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from abc import ABC, abstractmethod
from typing import Tuple, Literal, Dict, Type

import numpy as np

import heat1d_cpp
import wave1d_cpp

# 2D はまだ実装途中なので、モジュールが無くても落ちないようにしておく
try:
    import heat2d_cpp
except ImportError:
    heat2d_cpp = None


# ===============================
# type aliases
# ===============================

ICTypeStr = Literal["gaussian", "sine", "twopeaks"]
SolverKey = str


# ===============================
# Base classes
# ===============================

class ReferenceSolver(ABC):
    """
    すべての「参照ソルバ」の基底クラス。
    次元に依らず共通のインターフェースをここで定義する。
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def solve(self):
        """
        問題設定に応じた「参照解」を返す。
        具体的な戻り値の形は派生クラスごとに決まる。
        """
        raise NotImplementedError


class ReferenceSolver1D(ReferenceSolver):
    """
    1 次元 PDE 用参照ソルバの基底クラス。

    戻り値:
        x : (Nx,)
        u : (Nx,)
    """

    @abstractmethod
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ReferenceSolver2D(ReferenceSolver):
    """
    2 次元 PDE 用参照ソルバの基底クラス。

    戻り値の想定:
        x : (Nx,)        1D x 座標
        y : (Ny,)        1D y 座標
        u : (Ny, Nx)     2D 格子上のスカラー場 u(x_i, y_j)
    """

    @abstractmethod
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


# ===============================
# Registry
# ===============================

# 今後 1D/2D/3D すべてのソルバをここに登録していく
_SOLVER_REGISTRY: Dict[SolverKey, Type[ReferenceSolver]] = {}


def register_reference_solver(key: SolverKey):
    """
    参照ソルバクラスをレジストリに登録するためのデコレータ。

    例:
        @register_reference_solver("heat1d")
        class HeatReferenceSolver(ReferenceSolver1D):
            ...
    """

    def decorator(cls: Type[ReferenceSolver]) -> Type[ReferenceSolver]:
        _SOLVER_REGISTRY[key] = cls
        return cls

    return decorator


# ===============================
# Heat Reference Solver (1D)
# ===============================

@register_reference_solver("heat1d")
class HeatReferenceSolver(ReferenceSolver1D):
    """
    1D Heat equation reference solver.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.L = float(config["L"])
        self.alpha = float(config["alpha"])
        self.T_final = float(config["T_final"])

        self.Nx_cpp = int(config.get("Nx_cpp", 101))
        self.dt_cpp = float(config.get("dt_cpp", 5e-4))

        self.ic_type: ICTypeStr = config.get("ic_type", "gaussian").lower()
        self.gaussian_k: float = float(config.get("gaussian_k", 100.0))

    def _to_ic_enum(self):
        """ic_type(str) → heat1d_cpp.ICType への変換"""
        ICType = heat1d_cpp.ICType

        mapping = {
            "gaussian": ICType.Gaussian,
            "sine": ICType.Sine,
            "twopeaks": ICType.TwoPeaks,
        }
        return mapping.get(self.ic_type, ICType.Gaussian)

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        steps = int(self.T_final / self.dt_cpp)
        L = self.L

        # ==== IC = sine のときは解析解 ====
        if self.ic_type == "sine":
            x = np.linspace(0.0, L, self.Nx_cpp, dtype=float)
            factor = np.exp(-(np.pi ** 2) * self.alpha * self.T_final)
            u = np.sin(np.pi * x / L) * factor
            return x, u

        # ==== それ以外は C++ の solve_heat_1d を呼ぶ ====
        ic_enum = self._to_ic_enum()

        x_cpp, u_cpp = heat1d_cpp.solve_heat_1d(
            Nx=self.Nx_cpp,
            L=self.L,
            alpha=self.alpha,
            dt=self.dt_cpp,
            steps=steps,
            ic_type=ic_enum,
            gaussian_k=self.gaussian_k,
        )

        return np.asarray(x_cpp, dtype=float), np.asarray(u_cpp, dtype=float)


# ===============================
# Wave Reference Solver (1D)
# ===============================

@register_reference_solver("wave1d")
class WaveReferenceSolver(ReferenceSolver1D):
    """
    1D Wave equation reference solver.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.L = float(config["L"])
        self.T_final = float(config["T_final"])

        # unify names with GUI backend
        self.Nx_cpp = int(config.get("Nx_cpp", 201))
        self.c = float(config.get("c", 1.0))
        self.dt_cpp = float(config.get("dt_cpp", 0.001))

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        steps = int(self.T_final / self.dt_cpp)

        x_cpp, u_cpp = wave1d_cpp.solve_wave_1d(
            Nx=self.Nx_cpp,
            L=self.L,
            c=self.c,
            dt=self.dt_cpp,
            steps=steps,
        )

        return np.asarray(x_cpp, dtype=float), np.asarray(u_cpp, dtype=float)


# ===============================
# Heat Reference Solver (2D) - テンプレート
# ===============================

@register_reference_solver("heat2d")
class Heat2DReferenceSolver(ReferenceSolver2D):
    """
    2D Heat equation reference solver using C++ backend.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # PDE / Domain
        self.Nx = int(config.get("Nx_cpp", 101))
        self.Ny = int(config.get("Ny_cpp", 101))
        self.Lx = float(config.get("Lx", config.get("L", 1.0)))
        self.Ly = float(config.get("Ly", config.get("L", 1.0)))
        self.alpha = float(config.get("alpha", 0.01))
        self.dt = float(config.get("dt_cpp", 1e-4))
        self.T_final = float(config.get("T_final", 0.1))

        # IC
        self.ic_type = config.get("ic_type", "gaussian").lower()
        self.gaussian_kx = float(config.get("gaussian_kx", config.get("gaussian_k", 100.0)))
        self.gaussian_ky = float(config.get("gaussian_ky", config.get("gaussian_k", 100.0)))

    def _to_ic_enum(self):
        IC = heat2d_cpp.ICType2D
        mapping = {
            "gaussian": IC.Gaussian,
            "sine": IC.SineXY,
            "sinexy": IC.SineXY,
        }
        return mapping.get(self.ic_type, IC.Gaussian)

    def solve(self):
        steps = int(self.T_final / self.dt)

        x, y, u_flat = heat2d_cpp.solve_heat_2d(
            Nx=self.Nx,
            Ny=self.Ny,
            Lx=self.Lx,
            Ly=self.Ly,
            alpha=self.alpha,
            dt=self.dt,
            steps=steps,
            ic_type=self._to_ic_enum(),
            gaussian_kx=self.gaussian_kx,
            gaussian_ky=self.gaussian_ky,
        )

        if heat2d_cpp is None:
            raise ImportError("heat2d_cpp module is not available. Build it first.")


        return (
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(u_flat, dtype=float).reshape(self.Ny, self.Nx),
        )


# ===============================
# Factory
# ===============================

def make_reference_solver(config: dict, solver_type: SolverKey = "heat1d") -> ReferenceSolver:
    """
    solver_type と config から適切な ReferenceSolver を生成する。

    現在:
        - "heat1d"  -> HeatReferenceSolver (1D)
        - "wave1d"  -> WaveReferenceSolver (1D)
        - "heat2d"  -> Heat2DReferenceSolver (2D, テンプレート)
    """
    key = solver_type or config.get("solver_type", "heat1d")

    cls = _SOLVER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown reference solver type: {key}")

    return cls(config)
