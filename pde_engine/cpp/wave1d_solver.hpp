#pragma once
#include <vector>

/// 1D wave equation solver
/// u_tt = c^2 u_xx, Dirichlet BC u(0,t) = u(L,t) = 0
/// 初期条件:
///   u(x,0)   = sin(pi x / L)
///   u_t(x,0) = 0
class Wave1DSolver {
public:
    /// @param Nx_ 空間分割数
    /// @param L_  区間長さ [0, L]
    /// @param c_  波の速さ
    /// @param dt_ 時間刻み
    Wave1DSolver(int Nx_, double L_, double c_, double dt_);

    /// 現在のパラメータに応じた初期条件を設定
    void reset_initial();

    /// 1ステップ (dt 分) 時間発展
    void step();

    /// step() を steps 回繰り返す (t ≒ steps * dt まで進める)
    void run(int steps);

    /// 空間座標配列 x_i を返す
    const std::vector<double>& get_x() const { return x; }

    /// 現在の変位 u(x, t) を返す
    const std::vector<double>& get_u() const { return u_curr; }

    // 参考情報用 getter（必要なら Python 側からも利用可）
    int    get_Nx() const { return Nx; }
    double get_L()  const { return L; }
    double get_c()  const { return c; }
    double get_dt() const { return dt; }

private:
    int    Nx;
    double L;
    double c;
    double dt;
    double dx;       // 空間刻み
    double lambda2;  // (c dt / dx)^2

    bool first_step; // 1ステップ目かどうか

    std::vector<double> x;      // グリッド座標
    std::vector<double> u_prev; // u^{n-1}
    std::vector<double> u_curr; // u^{n}
    std::vector<double> u_next; // u^{n+1} 一時バッファ

    void update_coef();         // lambda2 を更新
};
