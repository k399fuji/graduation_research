# python/backend/wave_pinn_backend.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

import torch

import pinn_wave1d  # 波動方程式用 PINN 本体
from backend import log_utils

LOG_DIR = log_utils.LOG_DIR

# ============================
# Config / Result dataclass
# ============================

@dataclass
class WavePINNConfig:
    # PDE
    L: float = 1.0          # domain length
    c: float = 1.0          # wave speed
    T_final: float = 1.0    # final time

    # C++ solver
    Nx_cpp: int = 201
    dt_cpp: float = 0.001

    # PINN model
    hidden_dim: int = 64
    num_layers: int = 4
    lr: float = 1e-3
    epochs: int = 5000

    # sampling
    N_r: int = 2000
    N_ic_u: int = 200      # for u(x,0)
    N_ic_ut: int = 200     # for u_t(x,0)
    N_bc: int = 200

    # loss weights
    w_pde: float = 1.0
    w_ic_u: float = 1.0
    w_ic_ut: float = 1.0
    w_bc: float = 1.0

    # IC shape
    ic_type: str = "gaussian"   # "gaussian" / "sine" / "twopeaks"
    gaussian_k: float = 100.0

    # logging
    log_interval: int = 500
    tag: str = "qt"


@dataclass
class WavePINNResult:
    run_id: str
    l2_error: Optional[float]
    linf_error: Optional[float]
    log_csv_path: Optional[Path] = None
    config_json_path: Optional[Path] = None
    eval_json_path: Optional[Path] = None
    summary_json_path: Optional[Path] = None
    model_path: Optional[Path] = None


# ============================
# internal helpers
# ============================

def _config_to_dict(cfg: WavePINNConfig) -> dict:
    d = {
        "L": cfg.L,
        "c": cfg.c,
        "T_final": cfg.T_final,
        "Nx_cpp": cfg.Nx_cpp,
        "dt_cpp": cfg.dt_cpp,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "N_r": cfg.N_r,
        "N_ic_u": cfg.N_ic_u,
        "N_ic_ut": cfg.N_ic_ut,
        "N_bc": cfg.N_bc,
        "w_pde": cfg.w_pde,
        "w_ic_u": cfg.w_ic_u,
        "w_ic_ut": cfg.w_ic_ut,
        "w_bc": cfg.w_bc,
        "ic_type": cfg.ic_type,
        "gaussian_k": cfg.gaussian_k,
        "log_interval": cfg.log_interval,
        "tag": cfg.tag,
    }
    d["steps_cpp"] = int(d["T_final"] / d["dt_cpp"])
    
    dx = cfg.L / (cfg.Nx_cpp - 1)
    r = cfg.c * cfg.dt_cpp / dx
    print(f"[Wave] CFL number r = {r:.3f} (<= 1 なら安定)")

    return d


def _write_run_summary(
    run_id: str,
    config_dict: dict,
    l2: Optional[float],
    linf: Optional[float],
    log_csv: Path,
    config_path: Path,
    eval_path: Optional[Path],
    model_path: Optional[Path],
) -> Path:
    """
    wave1d 用の run_summary.json を生成
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / f"{run_id}_summary.json"

    solution_block = None
    if eval_path is not None and eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
            if all(k in ev for k in ("x_cpp", "u_cpp", "x_pinn", "u_pinn")):
                solution_block = {
                    "x_cpp": ev["x_cpp"],
                    "u_cpp": ev["u_cpp"],
                    "x_pinn": ev["x_pinn"],
                    "u_pinn": ev["u_pinn"],
                }
            if "L2_error" in ev and l2 is None:
                l2 = float(ev["L2_error"])
            if "Linf_error" in ev and linf is None:
                linf = float(ev["Linf_error"])
        except Exception:
            solution_block = None

    now_iso = datetime.now().astimezone().isoformat()

    run_block = {
        "run_id": run_id,
        "module": "wave1d",
        "backend": "pinn_wave1d",
        "created_at": now_iso,
        "tag": config_dict.get("tag", ""),
    }

    problem_block = {
        "pde_type": "wave",
        "dim": 1,
        "domain": {
            "x": [0.0, float(config_dict.get("L", 1.0))],
        },
        "c": float(config_dict.get("c", 1.0)),
        "T_final": float(config_dict.get("T_final", 1.0)),
        "initial_condition": {
            "type": config_dict.get("ic_type", "gaussian"),
            "params": {
                "gaussian_k": float(config_dict.get("gaussian_k", 100.0))
            },
        },
        "boundary_condition": {
            "type": "dirichlet_zero",
            "params": {},
        },
    }

    training_block = {
        "pinn": {
            "hidden_dim": int(config_dict.get("hidden_dim", 64)),
            "num_layers": int(config_dict.get("num_layers", 4)),
            "lr": float(config_dict.get("lr", 1e-3)),
            "epochs": int(config_dict.get("epochs", 5000)),
            "w_pde": float(config_dict.get("w_pde", 1.0)),
            "w_ic_u": float(config_dict.get("w_ic_u", 1.0)),
            "w_ic_ut": float(config_dict.get("w_ic_ut", 1.0)),
            "w_bc": float(config_dict.get("w_bc", 1.0)),
        },
        "sampling": {
            "N_r": int(config_dict.get("N_r", 0)),
            "N_ic_u": int(config_dict.get("N_ic_u", 0)),
            "N_ic_ut": int(config_dict.get("N_ic_ut", 0)),
            "N_bc": int(config_dict.get("N_bc", 0)),
        },
        "logging": {
            "log_interval": int(config_dict.get("log_interval", 1)),
            "log_csv_path": str(log_csv),
            "config_json_path": str(config_path),
            "eval_json_path": str(eval_path) if eval_path is not None else None,
        },
    }

    if model_path is not None:
        training_block["checkpoints"] = {
            "model_path": str(model_path)
        }

    results_block = {
        "metrics": {
            "L2_error": float(l2) if l2 is not None else None,
            "Linf_error": float(linf) if linf is not None else None,
        }
    }
    if solution_block is not None:
        results_block["solution_1d"] = solution_block

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

def run_wave_pinn(cfg: WavePINNConfig, progress_callback=None) -> WavePINNResult:
    """
    Wave PINN の学習〜評価をまとめて実行し、
    CSV / config.json / eval.json / run_summary.json を保存して結果を返す。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_dict = _config_to_dict(cfg)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"wave1d_{now_str}_{cfg.tag}"
    config_dict["run_id"] = run_id

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = log_utils.path_config(run_id)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # model / train / eval
    model = pinn_wave1d.build_model(config_dict, device)
    pinn_wave1d.train(model, config_dict, device, progress_callback=progress_callback)
    l2, linf = pinn_wave1d.evaluate(
        model, config_dict, device, make_plot=False, save_eval=True
    )

    model_path = log_utils.path_model(run_id)
    torch.save(
        {
            "config": config_dict,
            "state_dict": model.state_dict(),
        },
        model_path,
    )

    log_csv = log_utils.path_csv(run_id)
    eval_json = log_utils.path_eval(run_id)

    summary_path = _write_run_summary(
        run_id=run_id,
        config_dict=config_dict,
        l2=l2,
        linf=linf,
        log_csv=log_csv,
        config_path=config_path,
        eval_path=eval_json if eval_json.exists() else None,
        model_path=model_path if model_path.exists() else None,
    )

    return WavePINNResult(
        run_id=run_id,
        l2_error=float(l2),
        linf_error=float(linf),
        log_csv_path=log_csv if log_csv.exists() else None,
        config_json_path=config_path if config_path.exists() else None,
        eval_json_path=eval_json if eval_json.exists() else None,
        summary_json_path=summary_path if summary_path.exists() else None,
        model_path=model_path if model_path.exists() else None,
    )
