#include "methods.hh"
#include "lp_workspace_highs.hh"


#include <Eigen/Dense>
#include <numeric>
#include <memory>
#include <algorithm>
#include <functional>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <sstream>
#include <mutex>
#include <atomic>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using namespace m_height_lp;
struct JiangHighsJob {
    int a, b = -1;
    std::vector<int> X, Y;
    unsigned long long mask = 0;
};

double simplified_highs(const LpInput& G, int m, double threshold, int num_threads) {
    validate_highs_input(G, m, num_threads, threshold);
    if (m < 1 || m > 30) throw std::invalid_argument("Simplified Jiang requires 1 <= m <= 30.");
    const int k = G.rows(), n = G.cols();
    int a = 0;
    Combinations combinations(n - 1, m - 1);
    auto next = [&](JiangHighsJob& job) {
        if (a == n) return false;
        job.a = a;
        job.X.clear(); job.Y.clear();
        for (int j = 0; j < n - 1; ++j) {
            const int coordinate = j < a ? j : j + 1;
            (std::binary_search(combinations.values.begin(), combinations.values.end(), j)
                ? job.X : job.Y).push_back(coordinate);
        }
        if (!combinations.advance()) { ++a; combinations = Combinations(n - 1, m - 1); }
        return true;
    };
    auto evaluate = [&](HighsWorkspace& w, const JiangHighsJob& job, const auto&) {
        for (int c = 0; c < k; ++c) w.lp.col_cost_[c] = G(c, job.a);
        int row = 0;
        for (int x : job.X) {
            for (int c = 0; c < k; ++c) w.coefficient(row, c) = G(c, x) - G(c, job.a);
            w.lp.row_upper_[row++] = 0.0;
        }
        for (int y : job.Y) {
            for (int c = 0; c < k; ++c) {
                w.coefficient(row, c) = G(c, y);
                w.coefficient(row + 1, c) = -G(c, y);
            }
            w.lp.row_upper_[row++] = 1.0;
            w.lp.row_upper_[row++] = 1.0;
        }
        return primal_value(w.solve());
    };
    return highs_sweep<JiangHighsJob>(m - 1 + 2 * (n - m), k, true,
                                      num_threads, threshold, next, evaluate);
}

double original_highs(const LpInput& G, int m, int num_threads) {
    validate_highs_input(G, m, num_threads, std::numeric_limits<double>::infinity());
    if (m < 1 || m > 30) throw std::invalid_argument("Original Jiang requires 1 <= m <= 30.");
    const int k = G.rows(), n = G.cols();
    for (int j = 0; j < n; ++j)
        if (G.col(j).isZero(0.0)) throw std::invalid_argument("Original Jiang assumes no zero column.");
    int a = 0, b = 1;
    unsigned long long mask = 0;
    Combinations combinations(n - 2, m - 1);
    auto next = [&](JiangHighsJob& job) {
        if (a == n) return false;
        job.a = a; job.b = b; job.mask = mask;
        job.X.clear(); job.Y.clear();
        int position = 0;
        for (int j = 0; j < n; ++j) {
            if (j == a || j == b) continue;
            (std::binary_search(combinations.values.begin(), combinations.values.end(), position)
                ? job.X : job.Y).push_back(j);
            ++position;
        }
        if (++mask == (1ULL << m)) {
            mask = 0;
            if (!combinations.advance()) {
                combinations = Combinations(n - 2, m - 1);
                do { ++b; } while (b == a);
                if (b >= n) { ++a; b = a == 0 ? 1 : 0; }
            }
        }
        return true;
    };
    auto evaluate = [&](HighsWorkspace& w, const JiangHighsJob& job, const auto&) {
        const double sa = (job.mask & 1ULL) ? 1.0 : -1.0;
        for (int c = 0; c < k; ++c) w.lp.col_cost_[c] = sa * G(c, job.a);
        int row = 0;
        for (int t = 0; t < static_cast<int>(job.X.size()); ++t) {
            const double sx = (job.mask & (1ULL << (t + 1))) ? 1.0 : -1.0;
            for (int c = 0; c < k; ++c) {
                w.coefficient(row, c) = sx * G(c, job.X[t]) - sa * G(c, job.a);
                w.coefficient(row + 1, c) = -sx * G(c, job.X[t]);
            }
            w.lp.row_upper_[row++] = 0.0;
            w.lp.row_upper_[row++] = -1.0;
        }
        for (int c = 0; c < k; ++c) w.coefficient(row, c) = G(c, job.b);
        w.lp.row_lower_[row] = 1.0;
        w.lp.row_upper_[row++] = 1.0;
        for (int y : job.Y) {
            for (int c = 0; c < k; ++c) {
                w.coefficient(row, c) = G(c, y);
                w.coefficient(row + 1, c) = -G(c, y);
            }
            w.lp.row_upper_[row++] = 1.0;
            w.lp.row_upper_[row++] = 1.0;
        }
        const auto result = w.solve();
        if (result.status == Status::Optimal) return result.value;
        if (result.status == Status::Unbounded) return std::numeric_limits<double>::infinity();
        return -std::numeric_limits<double>::infinity(); // infeasible sign case
    };
    return highs_sweep<JiangHighsJob>(2 * (m - 1) + 1 + 2 * (n - m - 1), k,
        true, num_threads, std::numeric_limits<double>::infinity(), next, evaluate);
}
} // namespace

double h_m_jiang_simplified_lp_highs_early_quit(
    const LpInput& G, int m, double early_quit_threshold, int num_threads) {
    return simplified_highs(G, m, early_quit_threshold, num_threads);
}
double h_m_jiang_original_lp_highs(const LpInput& G, int m, int num_threads) {
    return original_highs(G, m, num_threads);
}
