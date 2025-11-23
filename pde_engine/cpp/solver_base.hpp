#pragma once

#include <vector>

// 1次元スカラー場 PDE ソルバの共通インターフェイス
class ISolver1D {
public:
    virtual ~ISolver1D() = default;

    // 初期条件をセット／リセット
    virtual void reset_initial() = 0;

    // 1ステップ分 時間発展
    virtual void step() = 0;

    // 共通の run 実装（必要ステップ数だけ step() を回す）
    virtual void run(int steps) {
        for (int n = 0; n < steps; ++n) {
            step();
        }
    }

    // 空間グリッドと現在の解 u(x, t) へのアクセス
    virtual const std::vector<double>& get_x() const = 0;
    virtual const std::vector<double>& get_u() const = 0;
};
