#pragma once
#include <vector>

/// 初期条件の種類
enum class ICType {
    Gaussian,  // ガウス分布 1 山
    Sine,      // sin(pi x / L)
    TwoPeaks   // ガウス 2 山
};

/// 1D 熱方程式 u_t = alpha * u_xx の陽解法ソルバ
class Heat1DSolver {
public:
    /// @param Nx_    空間分割数
    /// @param L_     区間長さ [0, L]
    /// @param alpha_ 拡散係数
    /// @param dt_    時間刻み
    Heat1DSolver(int Nx_, double L_, double alpha_, double dt_);

    /// 初期条件の種類とガウス係数を設定して、内部状態を初期化
    void set_initial_condition(ICType type, double gaussian_k_ = 100.0);

    /// 現在の設定(ic_type, gaussian_k, L) に基づいて u(x,0) を再生成
    void reset_initial();

    /// 1 ステップ時間発展（dt だけ進める）
    void step();

    /// step() を steps 回まわして T_final ≒ steps * dt まで進める
    void run(int steps);

    /// 空間座標配列を取得
    const std::vector<double>& get_x() const { return x; }

    /// 温度分布 u(x, t) を取得
    const std::vector<double>& get_u() const { return u; }

    // 参考用の getter（必要なら Python からも使える）
    int    get_Nx()    const { return Nx; }
    double get_L()     const { return L; }
    double get_alpha() const { return alpha; }
    double get_dt()    const { return dt; }

private:
    int    Nx;
    double L;
    double alpha;
    double dt;
    double dx;
    double coef;       // alpha * dt / dx^2

    ICType ic_type;    // 初期条件の種類
    double gaussian_k; // ガウスの「鋭さ」

    std::vector<double> x;
    std::vector<double> u;
    std::vector<double> un;

    /// x_val における u(x,0) を計算
    double initial_condition(double x_val) const;

    /// coef = alpha * dt / dx^2 を再計算
    void update_coef();
};
