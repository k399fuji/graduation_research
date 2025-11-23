import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================
#  パラメータ設定
# ============================
L = 1.0          # 空間長さ [0, L]
Nx = 101         # 空間分割数
dx = L / (Nx - 1)

alpha = 0.01     # 熱拡散係数
dt = 0.0005      # 時間刻み
steps = 800      # シミュレーションステップ数

# CFL条件: alpha * dt / dx^2 <= 1/2 を確認しておくと安全
cfl = alpha * dt / dx**2
print(f"CFL number = {cfl:.4f} (<= 0.5 が目安)")

# ============================
#  初期条件・配列の準備
# ============================
x = np.linspace(0.0, L, Nx)

# 初期条件: 真ん中あたりが熱い山の形 (ガウス分布)
u = np.exp(-((x - 0.5) ** 2) * 100.0)

# 結果表示用に図を準備
fig, ax = plt.subplots()
line, = ax.plot(x, u, lw=2)

ax.set_xlim(0, L)
ax.set_ylim(0, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("Temperature u(x,t)")
ax.set_title("1D Heat Equation Simulation")

# テキストで時間表示したい場合
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

# ============================
#  1ステップ分の更新関数
# ============================
def step(u: np.ndarray) -> np.ndarray:
    """
    熱方程式の陽的差分スキームで1ステップ時間発展させる。
    u: 現在時刻の温度分布 (1次元配列)
    return: 次の時刻の温度分布
    """
    un = u.copy()

    # 内部点のみ更新（i = 1 ... Nx-2）
    # u_i^{n+1} = u_i^n + alpha * dt / dx^2 * (u_{i-1}^n - 2u_i^n + u_{i+1}^n)
    coef = alpha * dt / dx**2
    for i in range(1, Nx - 1):
        un[i] = u[i] + coef * (u[i - 1] - 2.0 * u[i] + u[i + 1])

    # 境界条件: 両端の温度を0に固定（ディリクレ境界条件）
    un[0] = 0.0
    un[-1] = 0.0

    return un

# ============================
#  アニメーション用の更新関数
# ============================
current_step = 0

def update(frame):
    global u, current_step
    u = step(u)
    current_step += 1

    line.set_ydata(u)
    time_text.set_text(f"t = {current_step * dt:.4f}")

    return line, time_text

# ============================
#  アニメーション実行
# ============================
ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=30,   # ミリ秒 (表示の速さ)
    blit=True
)

plt.tight_layout()
plt.show()
