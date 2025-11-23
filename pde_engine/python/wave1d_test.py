import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 1つ上 = pde_engine をパスに追加

import matplotlib.pyplot as plt
import wave1d_cpp

# 関数版で試す
x, u_final = wave1d_cpp.solve_wave_1d(
    Nx=201,
    L=1.0,
    c=1.0,
    dt=0.003,
    steps=500
)

plt.figure()
plt.plot(x, u_final, label="Wave (function)")
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.title("1D Wave Equation (function wrapper)")
plt.legend()

# クラス版で試す
solver = wave1d_cpp.Wave1DSolver(
    Nx=201,
    L=1.0,
    c=1.0,
    dt=0.003
)
solver.reset_initial()
solver.run(500)

x2 = solver.get_x()
u2 = solver.get_u()

plt.figure()
plt.plot(x2, u2, label="Wave1DSolver (class)")
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.title("1D Wave Equation (class-based solver)")
plt.legend()

plt.show()
