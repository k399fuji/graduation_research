import os, sys

# このファイルの1つ上のディレクトリ（= pde_engine）をパスに追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import heat1d_cpp

# クラスとして利用
solver = heat1d_cpp.Heat1DSolver(
    Nx=101,
    L=1.0,
    alpha=0.01,
    dt=0.0005,
)

# 必要なら何度でも初期化できる
solver.reset_initial()

# 800ステップだけ時間発展
solver.run(800)

x = solver.get_x()
u = solver.get_u()

plt.plot(x, u, label="class-based solver")
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.title("Heat1DSolver result")
plt.legend()
plt.show()
