import numpy as np
import matplotlib.pyplot as plt

# C++ が出力した CSV を読み込む（パスは cpp/heat_cpp_final.csv）
data = np.loadtxt("../cpp/heat_cpp_final.csv", delimiter=",", skiprows=1)
x_cpp, u_cpp = data[:, 0], data[:, 1]

plt.plot(x_cpp, u_cpp, label="C++ final")
plt.xlabel("x")
plt.ylabel("u(x,T)")
plt.legend()
plt.title("C++ Heat Equation Final State")
plt.show()
