// heat2d_solver.hpp
#pragma once
#include <vector>

enum class ICType2D {
    Gaussian,  // 中央付近ガウシアン
    SineXY     // sin(πx/Lx) * sin(πy/Ly)
};

class Heat2DSolver {
public:
    Heat2DSolver(int Nx_, int Ny_,
                 double Lx_, double Ly_,
                 double alpha_, double dt_);

    void set_initial_condition(ICType2D type,
                               double gaussian_kx_ = 100.0,
                               double gaussian_ky_ = 100.0);

    void reset_initial();
    void step();
    void run(int steps);

    // グリッド情報
    int nx() const { return Nx; }
    int ny() const { return Ny; }
    double lx() const { return Lx; }
    double ly() const { return Ly; }

    const std::vector<double>& get_x() const { return x; }           // size Nx
    const std::vector<double>& get_y() const { return y; }           // size Ny
    const std::vector<double>& get_u() const { return u; }           // size Nx*Ny（row-major: j*Nx+i）

private:
    int Nx, Ny;
    double Lx, Ly;
    double alpha, dt;
    double dx, dy;
    double coef_x, coef_y;  // alpha*dt/dx^2, alpha*dt/dy^2

    ICType2D ic_type;
    double gaussian_kx, gaussian_ky;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> u;   // 現在の場（Ny*Nx）
    std::vector<double> un;  // 更新用バッファ

    void update_coef();
    double initial_condition(double x_val, double y_val) const;

    inline int idx(int i, int j) const { return j * Nx + i; } // i:0..Nx-1, j:0..Ny-1
};
