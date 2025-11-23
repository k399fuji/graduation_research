import os, sys

# このファイルの1つ上のディレクトリ（= pde_engine）をパスに追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import heat1d_cpp  # setup.py で作ったモジュール

# C++側の solver を呼ぶ
x, u_final = heat1d_cpp.solve_heat_1d(
    Nx=101,
    L=1.0,
    alpha=0.01,
    dt=0.0005,
    steps=800
)

# x, u_final は Python の list になっているので、そのままプロットできる
plt.plot(x, u_final, label="C++ via pybind11")
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.title("Heat equation solved by C++ (called from Python)")
plt.legend()
plt.show()
