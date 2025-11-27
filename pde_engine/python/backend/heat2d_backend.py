from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

import numpy as np

from backend import sim_log_utils as log_utils  # ★ ここは logs_sim 用のモジュールを使っている前提
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

    # アニメーションログ用
    save_anim: bool = True        # アニメーション用 npz を保存するか
    N_anim_frames: int = 40            # 将来: 保存したいフレーム数（今は未使用）

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
    anim_npz_path: Optional[Path] = None   # ★ 追加


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
        # アニメ用パラメータも保存しておく
        "save_anim": cfg.save_anim,
        "N_anim_frames": cfg.N_anim_frames,
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
    log_anim: Path | None = None,   # ★ 追加
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
            "anim_npz_path": str(log_anim) if log_anim is not None else None,  # ★ 追加
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


def _save_anim_npz(
    run_id: str,
    x: np.ndarray,
    y: np.ndarray,
    u2d: np.ndarray,
    T_final: float,
) -> Path:
    """
    アニメーション用の npz を保存する。
    現時点では「T_final の 1 フレームのみ」を保存しておき、
    将来的に multi-frame に拡張しやすい形にしておく。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 時刻軸: いまは 1 点だけ（将来は等間隔 t[0..Nt-1] にする）
    t = np.array([float(T_final)], dtype=float)

    # u の shape を (Nt, Ny, Nx) にそろえておく
    # ここでは Nt=1
    u_all = np.asarray(u2d, dtype=float)[np.newaxis, :, :]

    anim_path = LOG_DIR / f"{run_id}_anim.npz"
    np.savez(
        anim_path,
        t=t,
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        u=u_all,
    )
    return anim_path


# ============================
# main entry
# ============================

def run_heat2d(cfg: Heat2DConfig) -> Heat2DResult:
    """
    Heat2D の数値解を計算し、ログを保存して結果を返す。

    - eval.json / summary.json には T_final のスナップショットだけを保存
    - *_anim.npz には 0 < t_1 < ... < t_N = T_final の複数フレームを保存
    """

    # 1) dict 化
    config_dict = _config_to_dict(cfg)

    # 2) run_id 生成
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"heat2d_{now_str}_{cfg.tag}"
    config_dict["run_id"] = run_id

    # 3) config.json 保存（logs_sim 側）
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = log_utils.path_config(run_id)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # ========= 4) アニメーション用の時間サンプリング =========

    # N_anim_frames <= 1 のときは「T_final 1枚だけ」
    N_anim = max(1, int(cfg.N_anim_frames))
    T_final = float(cfg.T_final)
    dt_cpp = float(cfg.dt_cpp)

    if N_anim <= 1:
        # もとの挙動と同じ：T_final の 1 フレームだけ
        t_samples = np.array([T_final], dtype=float)
    else:
        # 0 を除いて (0, T_final] を等間隔サンプリング
        #   → 最初のフレームが dt_cpp より小さくなると step=0 になってしまうので、
        #      下限は dt_cpp に切り上げておく
        t_raw = np.linspace(dt_cpp, T_final, N_anim)
        t_samples = np.maximum(t_raw, dt_cpp)

    frames_u: list[np.ndarray] = []
    x_sample: Optional[np.ndarray] = None
    y_sample: Optional[np.ndarray] = None

    for t_target in t_samples:
        # このフレーム専用の config dict を作る
        frame_cfg = dict(config_dict)
        frame_cfg["T_final"] = float(t_target)
        frame_cfg["steps_cpp"] = int(t_target / dt_cpp)

        solver = make_reference_solver(frame_cfg, solver_type="heat2d")
        x, y, u2d = solver.solve()     # u2d: shape (Ny, Nx)

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        u_arr = np.asarray(u2d, dtype=float)

        if x_sample is None:
            x_sample = x_arr
            y_sample = y_arr
        frames_u.append(u_arr)

    # ここまでで
    #   x_sample: (Nx,)
    #   y_sample: (Ny,)
    #   frames_u: 長さ Nt のリスト（各要素 Ny×Nx）

    assert x_sample is not None and y_sample is not None
    u_all = np.stack(frames_u, axis=0)   # (Nt, Ny, Nx)
    t_array = np.asarray(t_samples, dtype=float)
    u_final = u_all[-1]                  # T_final に対応するフレーム

    # ========= 5) eval.json 保存（T_final のみ） =========

    eval_path = log_utils.path_eval(run_id)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "x": x_sample.tolist(),
                "y": y_sample.tolist(),
                "u": u_final.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========= 6) *_anim.npz 保存（複数フレーム） =========

    anim_path = LOG_DIR / f"{run_id}_anim.npz"
    np.savez(
        anim_path,
        x=x_sample,
        y=y_sample,
        t=t_array,
        u=u_all,
    )

    # ========= 7) summary.json 生成 =========

    summary_path = _write_run_summary(
        run_id=run_id,
        config_dict=config_dict,
        x=x_sample,
        y=y_sample,
        u=u_final,
        log_config=config_path,
        log_eval=eval_path if eval_path.exists() else None,
    )

    return Heat2DResult(
        run_id=run_id,
        x=x_sample,
        y=y_sample,
        u=u_final,
        config_json_path=config_path if config_path.exists() else None,
        eval_json_path=eval_path if eval_path.exists() else None,
        summary_json_path=summary_path if summary_path.exists() else None,
    )
