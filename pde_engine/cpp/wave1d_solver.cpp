// cpp/wave1d_solver.cpp
#include "wave1d_solver.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Wave1DSolver::Wave1DSolver(int Nx_, double L_, double c_, double dt_)
    : Nx(Nx_),
      L(L_),
      c(c_),
      dt(dt_),
      dx(0.0),
      lambda2(0.0),
      first_step(true),
      x(Nx_),
      u_prev(Nx_),
      u_curr(Nx_),
      u_next(Nx_)
{
    // グリッド幅と係数の計算
    dx = L / (Nx - 1);
    update_coef();

    // x 座標をセット
    for (int i = 0; i < Nx; ++i) {
        x[i] = dx * i;
    }

    // 初期条件をセット
    reset_initial();
}

void Wave1DSolver::update_coef() {
    double r = c * dt / dx;
    lambda2 = r * r;  // (c dt / dx)^2
}

// 初期条件:
//   u(x,0)   = sin(pi x / L)
//   u_t(x,0) = 0
void Wave1DSolver::reset_initial() {
    for (int i = 0; i < Nx; ++i) {
        double xi = x[i];
        u_curr[i] = std::sin(M_PI * xi / L);
    }

    // Dirichlet BC
    u_curr[0]      = 0.0;
    u_curr[Nx - 1] = 0.0;

    // 速度 u_t(x,0) = 0 のとき、
    // t = -dt の値として u_prev = u_curr を置くのが自然
    u_prev = u_curr;

    first_step = true;
}

void Wave1DSolver::step() {
    if (Nx < 3) return;

    if (first_step) {
        // --- 最初の一歩 t = 0 → dt ---
        // u^1_i = u^0_i + 0.5 * lambda2 * (u^0_{i-1} - 2u^0_i + u^0_{i+1})
        for (int i = 1; i < Nx - 1; ++i) {
            u_next[i] = u_curr[i]
                      + 0.5 * lambda2
                        * (u_curr[i - 1] - 2.0 * u_curr[i] + u_curr[i + 1]);
        }
        // 境界条件
        u_next[0]      = 0.0;
        u_next[Nx - 1] = 0.0;

        // 1ステップ進める
        u_prev = u_curr;
        u_curr = u_next;

        first_step = false;
        return;
    }

    // --- 通常ステップ ---
    // u^{n+1}_i = 2u^n_i - u^{n-1}_i + lambda2*(u^n_{i-1} - 2u^n_i + u^n_{i+1})
    for (int i = 1; i < Nx - 1; ++i) {
        u_next[i] = 2.0 * u_curr[i] - u_prev[i]
                  + lambda2
                    * (u_curr[i - 1] - 2.0 * u_curr[i] + u_curr[i + 1]);
    }

    // 境界条件
    u_next[0]      = 0.0;
    u_next[Nx - 1] = 0.0;

    // ステップ更新: n→n+1
    u_prev.swap(u_curr);
    u_curr.swap(u_next);
}

void Wave1DSolver::run(int steps) {
    for (int n = 0; n < steps; ++n) {
        step();
    }
}
