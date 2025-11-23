#include "heat1d_solver.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Heat1DSolver::Heat1DSolver(int Nx_, double L_, double alpha_, double dt_)
    : Nx(Nx_),
      L(L_),
      alpha(alpha_),
      dt(dt_),
      dx(0.0),
      coef(0.0),
      ic_type(ICType::Gaussian),
      gaussian_k(100.0),
      x(Nx_),
      u(Nx_),
      un(Nx_)  // バッファも Nx 分確保
{
    dx = L / (Nx - 1);
    update_coef();

    // x 座標を 0〜L に並べる
    for (int i = 0; i < Nx; ++i) {
        x[i] = dx * i;
    }

    // デフォルト初期条件で u, un をセット
    reset_initial();
}

void Heat1DSolver::set_initial_condition(ICType type, double gaussian_k_) {
    ic_type    = type;
    gaussian_k = gaussian_k_;
    reset_initial();
}

void Heat1DSolver::update_coef() {
    coef = alpha * dt / (dx * dx);
}

double Heat1DSolver::initial_condition(double x_val) const {
    switch (ic_type) {
    case ICType::Gaussian: {
        // 中心 0.5L のガウス
        double diff = x_val - 0.5 * L;
        return std::exp(-gaussian_k * diff * diff);
    }
    case ICType::Sine:
        // u(x,0) = sin(pi x / L)
        return std::sin(M_PI * x_val / L);

    case ICType::TwoPeaks: {
        // 中心 0.3L と 0.7L のガウス 2 山
        double c1 = 0.3 * L;
        double c2 = 0.7 * L;
        double d1 = x_val - c1;
        double d2 = x_val - c2;
        return std::exp(-gaussian_k * d1 * d1)
             + std::exp(-gaussian_k * d2 * d2);
    }

    default:
        return 0.0;
    }
}

void Heat1DSolver::reset_initial() {
    for (int i = 0; i < Nx; ++i) {
        u[i]  = initial_condition(x[i]);
        un[i] = u[i];  // バッファ側もそろえておく
    }
}

void Heat1DSolver::step() {
    // 内部点を陽解法で更新
    for (int i = 1; i < Nx - 1; ++i) {
        un[i] = u[i] + coef * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
    }

    // ディリクレ境界条件 u(0) = u(L) = 0
    un[0]      = 0.0;
    un[Nx - 1] = 0.0;

    u.swap(un);
}

void Heat1DSolver::run(int steps) {
    for (int n = 0; n < steps; ++n) {
        step();
    }
}
