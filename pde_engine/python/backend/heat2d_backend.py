# python/backend/heat2d_backend.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import json

import numpy as np

from backend import log_utils
from reference_solvers import make_reference_solver

LOG_DIR = log_utils.LOG_DIR


# ============================
# Config / Result dataclass
# ============================

@dataclass
class Heat2DConfig:
    # グリッド
    Nx_cpp: int = 101
    Ny_cpp: int = 101

    # 物理・領域
    Lx: float = 1.0
    Ly: float = 1.0
    alpha: float = 0.01
    T_final: float = 0.1
    dt_cpp: float = 1.0e-4

    # IC
    ic_type: str = "gaussian"   # "gaussian", "sine", "sinexy" など
    gaussian_kx: float = 100.0
    gaussian_ky: float = 100.0

    # その他
    tag: str = "heat2d"


@dataclass
class Heat2DResult:
    run_id: str
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    config_json_path: Optional[Path] = None
    eval_json_path: Optional[Path] = None
    summary_json_path: Optional[Path] = None


# ============================
# helpers
# ============================

def _config_to_dict(cfg: Heat2DConfig) -> dict:
    d = {
        "Nx_cpp": cfg.Nx_cpp,
        "Ny_cpp": cfg.Ny_cpp,
        "Lx": cfg.Lx,
        "Ly": cfg.Ly,
        "alpha": cfg.alpha,
        "T_final": cfg.T_final,
        "dt_cpp": cfg.dt_cpp,
        "ic_type": cfg.ic_type,
        "gaussian_kx": cfg.gaussian_kx,
        "gaussian_ky": cfg.gaussian_ky,
        "tag": cfg.tag,
        "solver_type": "heat2d",
    }
    d["steps_cpp"] = int(d["T_final"] / d["dt_cpp"])
    return d


def _write_run_summary(
    run_id: str,
    config_dict: dict,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    log_config: Path,
    log_eval: Path | None,
) -> Path:
    """
    heat2d 用 run_summary.json を作成
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / f"{run_id}_summary.json"

    now_iso = datetime.now().astimezone().isoformat()

    run_block = {
        "run_id": run_id,
        "module": "heat2d",
        "backend": "heat2d_backend",
        "created_at": now_iso,
        "tag": config_dict.get("tag", ""),
    }

    problem_block = {
        "pde_type": "heat",
        "dim": 2,
        "domain": {
            "x": [0.0, float(config_dict.get("Lx", 1.0))],
            "y": [0.0, float(config_dict.get("Ly", 1.0))],
        },
        "alpha": float(config_dict.get("alpha", 0.01)),
        "T_final": float(config_dict.get("T_final", 0.1)),
        "initial_condition": {
            "type": config_dict.get("ic_type", "gaussian"),
            "params": {
                "gaussian_kx": float(config_dict.get("gaussian_kx", 100.0)),
                "gaussian_ky": float(config_dict.get("gaussian_ky", 100.0)),
            },
        },
        "boundary_condition": {
            "type": "dirichlet_zero",
            "params": {},
        },
    }

    training_block = {
        "solver": {
            "type": "reference_2d",
            "Nx_cpp": int(config_dict.get("Nx_cpp", 0)),
            "Ny_cpp": int(config_dict.get("Ny_cpp", 0)),
            "dt_cpp": float(config_dict.get("dt_cpp", 0.0)),
        },
        "logging": {
            "config_json_path": str(log_config),
            "eval_json_path": str(log_eval) if log_eval is not None else None,
        },
    }

    results_block = {
        "metrics": {
            "L2_error": None,
            "Linf_error": None,
        },
        "solution_2d": {
            "x": x.tolist(),
            "y": y.tolist(),
            "u": u.tolist(),
        },
    }

    summary = {
        "schema_version": "0.1.0",
        "run": run_block,
        "problem": problem_block,
        "training": training_block,
        "results": results_block,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_path


# ============================
# main entry
# ============================

def run_heat2d(cfg: Heat2DConfig) -> Heat2DResult:
    """
    Heat2D の数値解を計算し、ログを保存して結果を返す。
    """
    # 1) dict 化
    config_dict = _config_to_dict(cfg)

    # 2) run_id 生成
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"heat2d_{now_str}_{cfg.tag}"
    config_dict["run_id"] = run_id

    # 3) config.json 保存
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = log_utils.path_config(run_id)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 4) C++ 参照ソルバで解く
    solver = make_reference_solver(config_dict, solver_type="heat2d")
    x, y, u2d = solver.solve()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u2d = np.asarray(u2d, dtype=float)

    # 5) eval.json 保存（可視化用）
    eval_path = log_utils.path_eval(run_id)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "x": x.tolist(),
                "y": y.tolist(),
                "u": u2d.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 6) summary 生成
    summary_path = _write_run_summary(
        run_id=run_id,
        config_dict=config_dict,
        x=x,
        y=y,
        u=u2d,
        log_config=config_path,
        log_eval=eval_path if eval_path.exists() else None,
    )

    return Heat2DResult(
        run_id=run_id,
        x=x,
        y=y,
        u=u2d,
        config_json_path=config_path if config_path.exists() else None,
        eval_json_path=eval_path if eval_path.exists() else None,
        summary_json_path=summary_path if summary_path.exists() else None,
    )
