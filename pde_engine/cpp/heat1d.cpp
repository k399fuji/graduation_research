#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>

int main() {
    // ============================
    // パラメータ（Python版と揃えてある）
    // ============================
    const double L = 1.0;        // 空間長さ [0, L]
    const int    Nx = 101;       // 空間分割数
    const double dx = L / (Nx - 1);

    const double alpha = 0.01;   // 熱拡散係数
    const double dt    = 0.0005; // 時間刻み
    const int    steps = 800;    // ステップ数（Pythonと同じ）

    const double cfl = alpha * dt / (dx * dx);
    std::cout << "CFL number = " << cfl << " (<= 0.5 が目安)\n";

    // ============================
    // メモリ確保
    // ============================
    std::vector<double> x(Nx);
    std::vector<double> u(Nx);
    std::vector<double> un(Nx);

    // 空間座標と初期条件（真ん中が熱いガウス分布）
    for (int i = 0; i < Nx; ++i) {
        x[i] = dx * i;
        double diff = x[i] - 0.5;
        u[i] = std::exp(- diff * diff * 100.0);
    }

    // ============================
    // 出力用ファイル（最終状態だけ書く）
    // ============================
    std::ofstream fout("heat_cpp_final.csv");
    if (!fout) {
        std::cerr << "ファイルを開けませんでした\n";
        return 1;
    }
    fout << "x,u\n";

    // ============================
    // 時間発展ループ
    // ============================
    const double coef = alpha * dt / (dx * dx);

    for (int n = 0; n < steps; ++n) {
        // 内部点のみ更新（i = 1 ... Nx-2）
        for (int i = 1; i < Nx - 1; ++i) {
            un[i] = u[i] + coef * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
        }

        // 境界条件: u(0) = u(L) = 0
        un[0]      = 0.0;
        un[Nx - 1] = 0.0;

        // u を更新
        u.swap(un);
    }

    // 最終状態をCSVに書き出す
    for (int i = 0; i < Nx; ++i) {
        fout << x[i] << "," << u[i] << "\n";
    }

    std::cout << "計算完了: heat_cpp_final.csv に出力しました\n";
    return 0;
}
