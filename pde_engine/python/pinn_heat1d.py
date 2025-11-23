import os
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
import argparse

# 1つ上(pde_engine)をパスに追加して C++ モジュールを import できるようにする
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from reference_solvers import make_reference_solver

# ============================
#  設定・パス関連
# ============================

from backend import log_utils

LOG_DIR = log_utils.LOG_DIR
LOG_DIR.mkdir(exist_ok=True)


def get_config():
    """デフォルト設定 + コマンドライン引数から config dict を生成"""
    parser = argparse.ArgumentParser(description="PINN for 1D Heat Equation")

    # PDE パラメータ
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--T_final", type=float, default=0.4)

    # C++ ソルバ側
    parser.add_argument("--Nx_cpp", type=int, default=101)
    parser.add_argument("--dt_cpp", type=float, default=0.0005)

    # PINN モデル
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5000)

    # サンプリング数
    parser.add_argument("--N_r", type=int, default=1000)
    parser.add_argument("--N_ic", type=int, default=200)
    parser.add_argument("--N_bc", type=int, default=200)

    # ログ周り
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--tag", type=str, default="default")

    args = parser.parse_args()

    config = {
        "L": args.L,
        "alpha": args.alpha,
        "T_final": args.T_final,
        "Nx_cpp": args.Nx_cpp,
        "dt_cpp": args.dt_cpp,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "epochs": args.epochs,
        "N_r": args.N_r,
        "N_ic": args.N_ic,
        "N_bc": args.N_bc,
        "log_interval": args.log_interval,
        "tag": args.tag,
    }

    # C++ ソルバのステップ数
    config["steps_cpp"] = int(config["T_final"] / config["dt_cpp"])

    # ランID（ログファイル名に使う）
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_id"] = f"heat1d_{now_str}_{config['tag']}"

    return config


# 初期条件
def initial_condition(x: torch.Tensor, config: dict) -> torch.Tensor:
    ic_type = config.get("ic_type", "gaussian")
    k = float(config.get("gaussian_k", 100.0))

    if ic_type == "gaussian":
        L = float(config.get("L", 1.0))
        return torch.exp(-((x - 0.5*L) ** 2) * k)


    elif ic_type == "sine":
        L = float(config.get("L", 1.0))
        return torch.sin(np.pi * x / L)

    elif ic_type == "twopeaks":
        L = float(config.get("L", 1.0))
        left = torch.exp(-((x - 0.3 * L) ** 2) * k)
        right = torch.exp(-((x - 0.7 * L) ** 2) * k)
        return left + right

    else:
        # fallback
        L = float(config.get("L", 1.0))
        return torch.exp(-((x - 0.5*L) ** 2) * k)




# ============================
#  PINN モデル定義
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
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def build_model(config, device):
    model = MLP(
        in_dim=2,
        out_dim=1,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)
    return model


# ============================
#  損失関数
# ============================

def pinn_loss(model, config, device):
    L = config["L"]
    alpha = config["alpha"]
    T_final = config["T_final"]
    N_r = config["N_r"]
    N_ic = config["N_ic"]
    N_bc = config["N_bc"]

    # --- PDE 残差用の点 ---
    x_r = torch.rand(N_r, 1, device=device) * L
    t_r = torch.rand(N_r, 1, device=device) * T_final
    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    u_r = model(x_r, t_r)
    ones = torch.ones_like(u_r)

    # u_t
    u_t = torch.autograd.grad(
        u_r, t_r, grad_outputs=ones,
        create_graph=True, retain_graph=True
    )[0]

    # u_x
    u_x = torch.autograd.grad(
        u_r, x_r, grad_outputs=ones,
        create_graph=True, retain_graph=True
    )[0]

    # u_xx
    u_xx = torch.autograd.grad(
        u_x, x_r, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]

    residual = u_t - alpha * u_xx
    loss_pde = (residual ** 2).mean()

    # --- 初期条件 ---
    x_ic = torch.rand(N_ic, 1, device=device) * L
    t_ic = torch.zeros(N_ic, 1, device=device)
    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = initial_condition(x_ic, config)
    loss_ic = ((u_ic_pred - u_ic_true) ** 2).mean()

    # --- 境界条件（両端固定） ---
    t_bc = torch.rand(N_bc, 1, device=device) * T_final
    x_left = torch.zeros(N_bc, 1, device=device)
    x_right = torch.ones(N_bc, 1, device=device) * L

    u_left = model(x_left, t_bc)
    u_right = model(x_right, t_bc)
    loss_bc = (u_left ** 2).mean() + (u_right ** 2).mean()

    w_pde = float(config.get("w_pde", 1.0))
    w_ic = float(config.get("w_ic", 1.0))
    w_bc = float(config.get("w_bc", 1.0))

    loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
    return loss, loss_pde, loss_ic, loss_bc


# ============================
#  学習ループ + ログ保存（改善版）
# ============================

# pinn_heat1d.py

def train(model, config, device, progress_callback=None):
    epochs = int(config.get("epochs", 5000))
    log_interval = int(config.get("log_interval", 500))
    run_id = config["run_id"]
    csv_path = LOG_DIR / f"{run_id}.csv"

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss, lpde, lic, lbc = pinn_loss(model, config, device)
        loss.backward()
        loss_total = float(loss.item())

        # 勾配ノルム（オプション）
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        grad_norm = float(grad_norm)

        optimizer.step()

        # ★ 進捗コールバック（Qt 用）
        if progress_callback is not None:
            try:
                progress_callback(epoch, loss_total)
            except Exception:
                # GUI 側でエラーが出ても学習は続ける
                pass

        # CSV ログ
        write_header = not csv_path.exists() or os.path.getsize(csv_path) == 0

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "loss_total", "loss_pde", "loss_ic", "loss_bc", "grad_norm"])
            writer.writerow([
                epoch,
                loss_total,
                float(lpde.item()),
                float(lic.item()),
                float(lbc.item()),
                grad_norm,
            ])

        # コンソールログ（お好みで）
        if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
            print(f"[{epoch}/{epochs}] loss={loss_total:.3e}")



# ============================
#  評価 & C++ 真値との比較
# ============================

# pinn_heat1d.py 内の evaluate をこの実装に置き換えてください

def evaluate(model, config, device, make_plot: bool = True, save_eval: bool = True):
    """
    PINN と C++ ソルバの解を比較し、
      - L2 / Linf 誤差の計算
      - eval.json への保存
      - 必要に応じてプロット
    を行う。

    C++ 側の詳細実装には依存せず、ReferenceSolver 抽象経由で参照解を取得する。
    """
    import numpy as np
    from pathlib import Path
    import json

    L = float(config["L"])
    T_final = float(config["T_final"])
    alpha = float(config.get("alpha", 0.01))        # ★ 追加
    ic_type = str(config.get("ic_type", "gaussian")).lower()  # ★ 追加

    # ============================
    # 1) PINN 側の解 u_pinn(x, T)
    # ============================
    Nx_eval = int(config.get("Nx_eval", 201))
    x_eval = torch.linspace(0.0, L, Nx_eval, device=device).view(-1, 1)
    t_eval = torch.ones_like(x_eval) * T_final

    with torch.no_grad():
        u_pinn_t = model(x_eval, t_eval).cpu().numpy().flatten()

    x_eval_np = x_eval.cpu().numpy().flatten()

    # ============================
    # 2) C++ 参照ソルバ（抽象経由）
    # ============================
    if ic_type == "sine":
        # 評価用グリッドを C++ 側とも揃えたいので Nx_cpp を利用
        Nx_cpp = int(config.get("Nx_cpp", 101))
        x_cpp = np.linspace(0.0, L, Nx_cpp)
        # u(x, T) = exp(-alpha * pi^2 * T) * sin(pi x)
        u_cpp = np.sin(np.pi * x_cpp) * np.exp(-alpha * (np.pi / L )** 2 * T_final)
    else:
        # 既存の C++ 参照ソルバを利用（gaussian など）
        ref_solver = make_reference_solver(config, solver_type="heat1d")
        x_cpp, u_cpp = ref_solver.solve()
        x_cpp = np.asarray(x_cpp, dtype=float)
        u_cpp = np.asarray(u_cpp, dtype=float)


    # C++ 解を PINN グリッドに補間して誤差計算
    u_cpp_interp = np.interp(x_eval_np, x_cpp, u_cpp)
    diff = u_pinn_t - u_cpp_interp

    l2 = float(np.sqrt(np.mean(diff ** 2)))
    linf = float(np.max(np.abs(diff)))

    print(f"[Evaluation at T = {T_final}]")
    print(f"  L2  error = {l2:.4e}")
    print(f"  Linf error = {linf:.4e}")

    # ============================
    # 3) eval.json 保存
    # ============================
    LOG_DIR.mkdir(exist_ok=True)

    eval_json_path = LOG_DIR / f"{config['run_id']}_eval.json"

    eval_data = {
        "run_id": config["run_id"],
        "L2_error": l2,
        "Linf_error": linf,
        "L": float(config.get("L", 1.0)),
        "alpha": float(config.get("alpha", 0.01)),
        "T_final": float(config.get("T_final", 0.4)),

        # 可視化用に生データも入れておく
        "x_cpp": x_cpp.tolist(),
        "u_cpp": u_cpp.tolist(),
        "x_pinn": x_eval_np.tolist(),
        "u_pinn": u_pinn_t.tolist(),
    }

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"評価ログを {eval_json_path.name} に保存しました。")

    # ============================
    # 4) プロット（オプション）
    # ============================
    if make_plot:
        plt.figure()
        plt.plot(x_cpp, u_cpp, "--", label="C++ reference")
        plt.plot(x_eval_np, u_pinn_t, label="PINN")
        plt.xlabel("x")
        plt.ylabel("u(x, T)")
        plt.title(f"Heat equation: C++ vs PINN (run_id={config['run_id']})")
        plt.legend()
        plt.show()

    return l2, linf

# ============================
#  メイン
# ============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    model = build_model(config, device)
    train(model, config, device)
    evaluate(model, config, device)


if __name__ == "__main__":
    main()
