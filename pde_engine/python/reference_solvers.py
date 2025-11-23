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


# ===============================
# type aliases
# ===============================

ICTypeStr = Literal["gaussian", "sine", "twopeaks"]
SolverKey = str


# ===============================
# Base class
# ===============================

class ReferenceSolver1D(ABC):
    """
    C++ ベース or 解析解ベースの「参照ソルバ」の共通インターフェース。
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        T_final まで時間発展させて (x, u(x, T_final)) を返す
        """
        raise NotImplementedError


# ===============================
# Registry
# ===============================

_SOLVER_REGISTRY: Dict[SolverKey, Type[ReferenceSolver1D]] = {}


def register_reference_solver(key: SolverKey):
    def decorator(cls: Type[ReferenceSolver1D]) -> Type[ReferenceSolver1D]:
        _SOLVER_REGISTRY[key] = cls
        return cls
    return decorator


# ===============================
# Heat Reference Solver
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
            # C++ 側に TwoPeaks が無いので、とりあえず Gaussian を使う
            "twopeaks": ICType.TwoPeaks,
        }
        return mapping.get(self.ic_type, ICType.Gaussian)

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        steps = int(self.T_final / self.dt_cpp)
        L = self.L

        # ==== IC = sine のときは解析解で OK ====
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
# Wave Reference Solver
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
# Factory
# ===============================

def make_reference_solver(config: dict, solver_type: SolverKey = "heat1d") -> ReferenceSolver1D:
    key = solver_type or config.get("solver_type", "heat1d")

    cls = _SOLVER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown reference solver type: {key}")

    return cls(config)
