// heat2d_solver.cpp
#include "heat2d_solver.hpp"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Heat2DSolver::Heat2DSolver(int Nx_, int Ny_,
                           double Lx_, double Ly_,
                           double alpha_, double dt_)
    : Nx(Nx_),
      Ny(Ny_),
      Lx(Lx_),
      Ly(Ly_),
      alpha(alpha_),
      dt(dt_),
      ic_type(ICType2D::Gaussian),
      gaussian_kx(100.0),
      gaussian_ky(100.0)
{
    dx = Lx / (Nx - 1);
    dy = Ly / (Ny - 1);
    update_coef();

    x.resize(Nx);
    y.resize(Ny);
    u.resize(Nx * Ny);
    un.resize(Nx * Ny);

    for (int i = 0; i < Nx; ++i) {
        x[i] = dx * i;
    }
    for (int j = 0; j < Ny; ++j) {
        y[j] = dy * j;
    }

    // 安定性チェック（参考）
    double cfl = coef_x + coef_y; // alpha*dt*(1/dx^2+1/dy^2)
    std::cout << "[Heat2D] CFL = " << cfl << " (<= 0.5 推奨)\n";

    reset_initial();
}

void Heat2DSolver::set_initial_condition(ICType2D type,
                                         double gaussian_kx_,
                                         double gaussian_ky_)
{
    ic_type     = type;
    gaussian_kx = gaussian_kx_;
    gaussian_ky = gaussian_ky_;
    reset_initial();
}

void Heat2DSolver::update_coef() {
    coef_x = alpha * dt / (dx * dx);
    coef_y = alpha * dt / (dy * dy);
}

double Heat2DSolver::initial_condition(double x_val, double y_val) const {
    switch (ic_type) {
    case ICType2D::Gaussian: {
        double cx = 0.5 * Lx;
        double cy = 0.5 * Ly;
        double dx_ = x_val - cx;
        double dy_ = y_val - cy;
        return std::exp(- (gaussian_kx * dx_ * dx_ + gaussian_ky * dy_ * dy_));
    }
    case ICType2D::SineXY: {
        double sx = std::sin(M_PI * x_val / Lx);
        double sy = std::sin(M_PI * y_val / Ly);
        return sx * sy;
    }
    default:
        return 0.0;
    }
}

void Heat2DSolver::reset_initial() {
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int k = idx(i, j);
            u[k] = initial_condition(x[i], y[j]);
        }
    }
}

void Heat2DSolver::step() {
    // 内部点を陽的差分で更新
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            int k = idx(i, j);
            int kL = idx(i - 1, j);
            int kR = idx(i + 1, j);
            int kD = idx(i, j - 1);
            int kU = idx(i, j + 1);

            double lap =
                (u[kL] - 2.0 * u[k] + u[kR]) / (dx * dx) +
                (u[kD] - 2.0 * u[k] + u[kU]) / (dy * dy);

            un[k] = u[k] + alpha * dt * lap;
        }
    }

    // Dirichlet 0 境界
    for (int i = 0; i < Nx; ++i) {
        un[idx(i, 0)]      = 0.0;
        un[idx(i, Ny - 1)] = 0.0;
    }
    for (int j = 0; j < Ny; ++j) {
        un[idx(0, j)]      = 0.0;
        un[idx(Nx - 1, j)] = 0.0;
    }

    u.swap(un);
}

void Heat2DSolver::run(int steps) {
    for (int n = 0; n < steps; ++n) {
        step();
    }
}
