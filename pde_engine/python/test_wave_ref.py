# python/test_wave_ref.py
import numpy as np
from reference_solvers import make_reference_solver

config = {
    "L": 1.0,
    "T_final": 1.0,
    "Nx_wave": 201,
    "wave_c": 1.0,
    "dt_wave": 0.001,
}

solver = make_reference_solver(config, solver_type="wave1d")
x, u = solver.solve()

print("x.shape =", x.shape)
print("u.shape =", u.shape)
print("u[0:5] =", u[:5])
