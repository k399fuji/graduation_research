# python/gui/app.py

from __future__ import annotations
from dataclasses import asdict

import os
import sys

# ★ ここを追加： app.py から見て 1つ上の "python" フォルダを sys.path に入れる
CURRENT_DIR = os.path.dirname(__file__)
PYTHON_DIR = os.path.dirname(CURRENT_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.append(PYTHON_DIR)

import streamlit as st

from backend.heat_pinn_backend import HeatPINNConfig, run_heat_pinn
from backend.log_utils import list_runs, load_config, load_eval, load_loss_csv


def page_heat_pinn():
    st.header("Heat Equation PINN")

    # ---- サイドバーで設定を入力 ----
    st.sidebar.subheader("Heat PINN Settings")

    L = st.sidebar.number_input("L (domain length)", value=1.0)
    alpha = st.sidebar.number_input("alpha (diffusivity)", value=0.01)
    T_final = st.sidebar.number_input("T_final", value=0.4)

    epochs = st.sidebar.number_input("epochs", min_value=100, max_value=50000, value=5000, step=100)
    hidden_dim = st.sidebar.number_input("hidden_dim", min_value=8, max_value=512, value=64, step=8)
    num_layers = st.sidebar.number_input("num_layers", min_value=2, max_value=10, value=4, step=1)
    lr = st.sidebar.number_input("learning rate", value=1e-3, format="%.1e")

    N_r = st.sidebar.number_input("N_r (residual points)", min_value=100, max_value=20000, value=1000, step=100)
    N_ic = st.sidebar.number_input("N_ic (IC points)", min_value=10, max_value=5000, value=200, step=10)
    N_bc = st.sidebar.number_input("N_bc (BC points)", min_value=10, max_value=5000, value=200, step=10)

    tag = st.sidebar.text_input("tag", value="gui")

    cfg = HeatPINNConfig(
        L=L,
        alpha=alpha,
        T_final=T_final,
        epochs=int(epochs),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        lr=float(lr),
        N_r=int(N_r),
        N_ic=int(N_ic),
        N_bc=int(N_bc),
        tag=tag,
    )

    st.write("Current config:", asdict(cfg))

    if st.button("Run Heat PINN experiment"):
        with st.spinner("Running Heat PINN..."):
            result = run_heat_pinn(cfg)

        st.success("Finished Heat PINN experiment!")
        st.write(f"run_id: `{result.run_id}`")

        if result.l2_error is not None:
            st.metric("L2 error", f"{result.l2_error:.3e}")
        if result.linf_error is not None:
            st.metric("L∞ error", f"{result.linf_error:.3e}")

        if result.log_csv_path and result.log_csv_path.exists():
            st.write("Log CSV:", result.log_csv_path.name)
        if result.config_json_path and result.config_json_path.exists():
            st.write("Config JSON:", result.config_json_path.name)
        if result.eval_json_path and result.eval_json_path.exists():
            st.write("Eval JSON:", result.eval_json_path.name)


def page_logs():
    st.header("Experiment Logs")

    runs = list_runs()
    if not runs:
        st.info("No runs found in logs/ yet.")
        return

    run_ids = [r.run_id for r in runs]
    selected_id = st.selectbox("Select run_id", run_ids)

    st.subheader("Config")
    cfg = load_config(selected_id)
    if cfg is None:
        st.write("No config.json found.")
    else:
        st.json(cfg)

    st.subheader("Evaluation")
    ev = load_eval(selected_id)
    if ev is None:
        st.write("No eval.json found yet.")
    else:
        st.json(ev)

    st.subheader("Loss curves")
    rows = load_loss_csv(selected_id)
    if not rows:
        st.write("No CSV log found.")
    else:
        # epoch vs loss_total だけ簡易表示（本格的には line chart にする）
        import pandas as pd
        df = pd.DataFrame(rows)
        df["epoch"] = df["epoch"].astype(int)

         # 古いログフォーマットへの対応：
         # "loss_total" が無くて "loss" だけある場合は、名前を揃える
        if "loss_total" not in df.columns and "loss" in df.columns:
            df.rename(columns={"loss": "loss_total"}, inplace=True)

        for col in ["loss_total", "loss_pde", "loss_ic", "loss_bc", "grad_norm"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 実際にプロットする列（存在するものだけ）
        cols = [c for c in ["loss_total", "loss_pde", "loss_ic", "loss_bc"] if c in df.columns]
        if cols:
            st.line_chart(df.set_index("epoch")[cols])
        else:
            st.write("プロット可能な loss 列が見つかりませんでした。")



def main():
    st.set_page_config(page_title="PDE Engine & PINN Dashboard", layout="wide")

    st.sidebar.title("PDE Engine Dashboard")
    page = st.sidebar.radio(
        "Page",
        ["Heat PINN", "Logs"],
    )

    if page == "Heat PINN":
        page_heat_pinn()
    elif page == "Logs":
        page_logs()


if __name__ == "__main__":
    main()
