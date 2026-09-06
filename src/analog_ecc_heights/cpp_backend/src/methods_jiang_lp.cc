#include "methods.hh"
#include "lp_workspace.hh"

#ifdef HAVE_HIGHS
#include <highs/Highs.h> 
#endif // HAVE_HIGHS

#include <glpk.h>
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

// Enumerate (a, X) lazily: X contains m-1 coordinates other than a,
// and Y is the complement of {a} union X. GLPK solves remain sequential.
static double jiang_simplified_glpk_value(
    const Eigen::MatrixXd& G, int m, double early_quit_threshold)
{
    const int k = G.rows(), n = G.cols();
    if (k == 0 || n == 0 || !G.allFinite())
        throw std::invalid_argument("G must be a finite, nonempty matrix.");
    if (m < 1 || m >= n || m > 30)
        throw std::invalid_argument("Simplified Jiang GLPK requires 1 <= m <= min(30, n-1).");
    if (std::isnan(early_quit_threshold))
        throw std::invalid_argument("early_quit_threshold must not be NaN.");

    m_height_lp::Workspace lp(m - 1 + 2 * (n - m), k,
                             m_height_lp::Backend::Glpk, true);
    lp.upper.setOnes();
    lp.upper.head(m - 1).setZero();
    double best = -std::numeric_limits<double>::infinity();
    bool stop = false;

    for (int a = 0; a < n && !stop; ++a) {
        lp.cost = G.col(a);
        std::vector<int> remaining, X;
        for (int j = 0; j < n; ++j)
            if (j != a) remaining.push_back(j);
        std::function<void(int, int)> enumerate = [&](int start, int need) {
            if (stop) return;
            if (need != 0) {
                for (int t = start; t <= static_cast<int>(remaining.size()) - need && !stop; ++t) {
                    X.push_back(remaining[t]);
                    enumerate(t + 1, need - 1);
                    X.pop_back();
                }
                return;
            }

            int row = 0;
            // c_x <= c_a for x in X.
            for (int x : X)
                lp.A.row(row++) = (G.col(x) - G.col(a)).transpose();
            // -1 <= c_y <= 1 for y in Y.
            for (int y : remaining) {
                if (std::binary_search(X.begin(), X.end(), y)) continue;
                lp.A.row(row++) = G.col(y).transpose();
                lp.A.row(row++) = -G.col(y).transpose();
            }

            const auto result = lp.solve();
            if (result.status == m_height_lp::Status::Optimal)
                best = std::max(best, result.value);
            else if (result.status == m_height_lp::Status::Unbounded)
                best = std::numeric_limits<double>::infinity();
            else
                throw std::runtime_error("Simplified Jiang LP unexpectedly infeasible (u=0 is feasible).");
            stop = best > early_quit_threshold || std::isinf(best);
        };
        enumerate(0, m - 1);
    }
    return std::min(best, early_quit_threshold);
}

double h_m_jiang_simplified_lp_glpk_early_quit(const Eigen::MatrixXd& G,
                                    int m,
                                    double early_quit_threshold) {
    return jiang_simplified_glpk_value(G, m, early_quit_threshold);
}

namespace {
using OriginalBackend = m_height_lp::Backend;

double jiang_original_lp(const Eigen::MatrixXd& G, int m, OriginalBackend backend) {
    const int k = G.rows(), n = G.cols();
    if (k == 0 || n < 2 || !G.allFinite())
        throw std::invalid_argument("G must be finite, nonempty, with at least two columns.");
    if (m < 1 || m >= n || m > 30)
        throw std::invalid_argument("Jiang original LP requires 1 <= m <= min(30, n-1).");
    for (int j = 0; j < n; ++j)
        if (G.col(j).isZero(0.0))
            throw std::invalid_argument("Jiang original LP assumes no zero column.");

    const int rows = 2 * (m - 1) + 1 + 2 * (n - m - 1);
    const int equality_row = 2 * (m - 1);
    m_height_lp::Workspace lp(rows, k, backend, true);
    auto& A = lp.A;
    auto& bounds = lp.upper;
    auto& cost = lp.cost;
    lp.lower[equality_row] = 1.0;
    double best = 0.0;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            std::vector<int> remaining, X;
            for (int j = 0; j < n; ++j)
                if (j != a && j != b) remaining.push_back(j);
            std::function<void(int, int)> enumerate = [&](int start, int need) {
                if (need != 0) {
                    for (int t = start; t <= static_cast<int>(remaining.size()) - need; ++t) {
                        X.push_back(remaining[t]);
                        enumerate(t + 1, need - 1);
                        X.pop_back();
                    }
                    return;
                }
                std::vector<int> Y;
                for (int j : remaining)
                    if (!std::binary_search(X.begin(), X.end(), j)) Y.push_back(j);
                for (unsigned long long mask = 0; mask < (1ULL << m); ++mask) {
                    const double sa = (mask & 1ULL) ? 1.0 : -1.0;
                    cost = sa * G.col(a);
                    int row = 0;
                    for (int t = 0; t < static_cast<int>(X.size()); ++t) {
                        const double sx = (mask & (1ULL << (t + 1))) ? 1.0 : -1.0;
                        // 1 <= sx*c_x <= sa*c_a.
                        A.row(row) = (sx * G.col(X[t]) - cost).transpose();
                        bounds[row++] = 0.0;
                        A.row(row) = (-sx * G.col(X[t])).transpose();
                        bounds[row++] = -1.0;
                    }
                    A.row(row) = G.col(b).transpose();
                    bounds[row++] = 1.0;
                    for (int j : Y) {
                        A.row(row) = G.col(j).transpose();
                        bounds[row++] = 1.0;
                        A.row(row) = -G.col(j).transpose();
                        bounds[row++] = 1.0;
                    }
                    const auto result = lp.solve();
                    if (result.status == m_height_lp::Status::Optimal)
                        best = std::max(best, result.value);
                    else if (result.status == m_height_lp::Status::Unbounded)
                        best = std::numeric_limits<double>::infinity();
                    // Infeasible sign/order cases contribute nothing.
                }
            };
            enumerate(0, m - 1);
        }
    }
    return best;
}
} // namespace

double h_m_jiang_original_lp_glpk(const Eigen::MatrixXd& G, int m) {
    return jiang_original_lp(G, m, OriginalBackend::Glpk);
}

#ifdef HAVE_HIGHS
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
#endif
