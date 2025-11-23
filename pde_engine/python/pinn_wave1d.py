# python/pinn_wave1d.py

import os
import sys
import json
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from reference_solvers import make_reference_solver
from backend import log_utils

LOG_DIR = log_utils.LOG_DIR
LOG_DIR.mkdir(exist_ok=True)


# ============================
# Initial Conditions
# ============================

def initial_condition_u(x: torch.Tensor, config: dict) -> torch.Tensor:
    """u(x,0)"""
    ic_type = config.get("ic_type", "gaussian")
    k = float(config.get("gaussian_k", 100.0))
    L = float(config.get("L", 1.0))

    if ic_type == "gaussian":
        return torch.exp(-((x - 0.5 * L)**2) * k)

    elif ic_type == "sine":
        return torch.sin(np.pi * x / L)

    elif ic_type == "twopeaks":
        left = torch.exp(-((x - 0.3 * L)**2) * k)
        right = torch.exp(-((x - 0.7 * LOG_DIR)**2) * k)
        return left + right

    return torch.exp(-((x - 0.5 * L)**2) * k)


def initial_condition_ut(x: torch.Tensor, config: dict) -> torch.Tensor:
    """u_t(x,0) は今は 0 とする。後で拡張可能"""
    return torch.zeros_like(x)



# ============================
# PINN Model
# ============================

class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))



def build_model(config, device):
    return MLP(
        in_dim=2,
        out_dim=1,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)



# ============================
# PINN Loss for Wave Equation
# ============================

def pinn_loss(model, config, device):
    L = config["L"]
    c = config["c"]
    T_final = config["T_final"]

    N_r = config["N_r"]
    N_ic_u = config["N_ic_u"]
    N_ic_ut = config["N_ic_ut"]
    N_bc = config["N_bc"]

    # ----- PDE residual u_tt - c^2 u_xx -----
    x_r = torch.rand(N_r, 1, device=device) * L
    t_r = torch.rand(N_r, 1, device=device) * T_final
    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    u_r = model(x_r, t_r)
    ones = torch.ones_like(u_r)

    # First derivatives
    u_t = torch.autograd.grad(u_r, t_r, grad_outputs=ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u_r, x_r, grad_outputs=ones, create_graph=True)[0]

    # Second derivatives
    u_tt = torch.autograd.grad(u_t, t_r, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_r, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = u_tt - (c**2) * u_xx
    loss_pde = (residual**2).mean()

    # ----- Initial condition: u(x,0) -----
    x_ic = torch.rand(N_ic_u, 1, device=device) * L
    t_ic = torch.zeros(N_ic_u, 1, device=device)

    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = initial_condition_u(x_ic, config)

    loss_ic_u = ((u_ic_pred - u_ic_true)**2).mean()

    # ----- Initial condition: u_t(x,0) -----
    x_ut = torch.rand(N_ic_ut, 1, device=device) * L
    t_ut = torch.zeros(N_ic_ut, 1, device=device)
    x_ut.requires_grad_(True)
    t_ut.requires_grad_(True)

    u_ut_pred = model(x_ut, t_ut)
    u_ut_t = torch.autograd.grad(
        u_ut_pred, t_ut, grad_outputs=torch.ones_like(u_ut_pred),
        create_graph=True
    )[0]

    u_ut_true = initial_condition_ut(x_ut, config)

    loss_ic_ut = ((u_ut_t - u_ut_true)**2).mean()

    # ----- Boundary (Dirichlet zero) -----
    t_bc = torch.rand(N_bc, 1, device=device) * T_final
    x_left = torch.zeros(N_bc, 1, device=device)
    x_right = torch.ones(N_bc, 1, device=device) * L

    u_left = model(x_left, t_bc)
    u_right = model(x_right, t_bc)

    loss_bc = (u_left**2).mean() + (u_right**2).mean()

    # ハイパー
    loss = (
        config["w_pde"] * loss_pde
        + config["w_ic_u"] * loss_ic_u
        + config["w_ic_ut"] * loss_ic_ut
        + config["w_bc"] * loss_bc
    )

    return loss, loss_pde, loss_ic_u, loss_ic_ut, loss_bc



# ============================
# Training Loop
# ============================

def train(model, config, device, progress_callback=None):
    epochs = config["epochs"]
    log_interval = int(config.get("log_interval", 500))
    run_id = config["run_id"]

    csv_path = LOG_DIR / f"{run_id}.csv"
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss, lpde, licu, licut, lbc = pinn_loss(model, config, device)
        loss.backward()
        optimizer.step()

        loss_total = float(loss.item())

        # 勾配ノルム（オプション）
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        grad_norm = float(grad_norm)

        # 進捗コールバック（Qt 用）
        if progress_callback is not None:
            try:
                progress_callback(epoch, loss_total)
            except Exception:
                # GUI 側で多少エラーが出ても学習は続行
                pass

        # CSV ログ
        write_header = not csv_path.exists() or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["epoch", "loss_total", "loss_pde", "loss_ic_u", "loss_ic_ut", "loss_bc", "grad_norm"]
                )
            writer.writerow(
                [
                    epoch,
                    loss_total,
                    float(lpde.item()),
                    float(licu.item()),
                    float(licut.item()),
                    float(lbc.item()),
                    grad_norm,
                ]
            )

        # コンソールログ
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            print(f"[Wave] [{epoch}/{epochs}] loss={loss_total:.3e}")


def evaluate(model, config, device, make_plot: bool = True, save_eval: bool = True):
    """
    Wave PINN と C++ 波動ソルバの解を比較し、
    L2 / Linf 誤差を計算して eval.json に保存する。

    eval.json のキーは heat と同じ
      x_cpp, u_cpp, x_pinn, u_pinn
    にしておくので、既存の Viewport(Solution / Error) でそのまま表示できます。
    """
    L = float(config["L"])
    T_final = float(config["T_final"])

    # 1) PINN 側の解 u_pinn(x, T_final)
    Nx_eval = int(config.get("Nx_eval", 201))
    x_eval = torch.linspace(0.0, L, Nx_eval, device=device).view(-1, 1)
    t_eval = torch.ones_like(x_eval) * T_final

    with torch.no_grad():
        u_pinn_t = model(x_eval, t_eval).cpu().numpy().flatten()

    x_eval_np = x_eval.cpu().numpy().flatten()

    # 2) C++ 参照ソルバ（波動方程式）
    ref_solver = make_reference_solver(config, solver_type="wave1d")
    x_cpp, u_cpp = ref_solver.solve()
    x_cpp = np.asarray(x_cpp, dtype=float)
    u_cpp = np.asarray(u_cpp, dtype=float)

    # 3) C++ 解を PINN グリッドに補間して誤差計算
    u_cpp_interp = np.interp(x_eval_np, x_cpp, u_cpp)
    diff = u_pinn_t - u_cpp_interp

    l2 = float(np.sqrt(np.mean(diff ** 2)))
    linf = float(np.max(np.abs(diff)))

    print(f"[Wave Evaluation at T = {T_final}]")
    print(f"  L2  error = {l2:.4e}")
    print(f"  Linf error = {linf:.4e}")

    # 4) eval.json 保存（heat と同じ形式）
    eval_json_path = LOG_DIR / f"{config['run_id']}_eval.json"
    eval_data = {
        "run_id": config["run_id"],
        "L2_error": l2,
        "Linf_error": linf,
        "L": float(config.get("L", 1.0)),
        "c": float(config.get("c", 1.0)),  # 波の速さがあれば
        "T_final": float(config.get("T_final", 1.0)),
        "x_cpp": x_cpp.tolist(),
        "u_cpp": u_cpp.tolist(),
        "x_pinn": x_eval_np.tolist(),
        "u_pinn": u_pinn_t.tolist(),
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"[Wave] 評価ログを {eval_json_path.name} に保存しました。")

    # 5) プロット（任意）
    if make_plot:
        plt.figure()
        plt.plot(x_cpp, u_cpp, "--", label="C++ reference")
        plt.plot(x_eval_np, u_pinn_t, label="Wave PINN")
        plt.xlabel("x")
        plt.ylabel("u(x, T)")
        plt.title(f"Wave equation: C++ vs PINN (run_id={config['run_id']})")
        plt.legend()
        plt.show()

    return l2, linf