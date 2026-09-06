#include "methods.hh"
#include "lp_workspace.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using m_height_lp::Backend;
using m_height_lp::Status;

double roth_lp(const Eigen::MatrixXd& G, int m, double threshold,
               Backend backend, bool primal) {
    const int k = G.rows(), n = G.cols();
    if (k == 0 || n == 0 || !G.allFinite())
        throw std::invalid_argument("G must be a finite, nonempty matrix.");
    if (m < 0 || m >= n)
        throw std::invalid_argument("m must satisfy 0 <= m <= n-1.");
    if (std::isnan(threshold))
        throw std::invalid_argument("early_quit_threshold must not be NaN.");
    if (m == 0) return std::min(1.0, threshold);

    const int q = n - m;
    m_height_lp::Workspace lp(primal ? 2 * q : k, primal ? k : 2 * q, backend, primal);
    if (primal) {
        lp.upper.setOnes();
    } else {
        lp.cost.setOnes();
        lp.col_lower.setZero();
    }

    double best = -std::numeric_limits<double>::infinity();
    bool stop = false;
    std::vector<int> S;
    std::function<void(int, int)> enumerate = [&](int start, int need) {
        if (stop) return;
        if (need != 0) {
            for (int j = start; j <= n - need && !stop; ++j) {
                S.push_back(j);
                enumerate(j + 1, need - 1);
                S.pop_back();
            }
            return;
        }
        std::vector<int> complement;
        for (int j = 0; j < n; ++j)
            if (!std::binary_search(S.begin(), S.end(), j)) complement.push_back(j);
        for (int t = 0; t < q; ++t) {
            if (primal) {
                // -1 <= u*g_j <= 1 on S complement.
                lp.A.row(2 * t) = G.col(complement[t]).transpose();
                lp.A.row(2 * t + 1) = -G.col(complement[t]).transpose();
            } else {
                // G_complement * (y-z) = g_i, y,z >= 0.
                lp.A.col(t) = G.col(complement[t]);
                lp.A.col(q + t) = -G.col(complement[t]);
            }
        }
        for (int i : S) {
            if (primal) lp.cost = G.col(i);
            else lp.lower = lp.upper = G.col(i);
            const auto result = lp.solve();
            double value;
            if (result.status == Status::Optimal) value = result.value;
            else if ((primal && result.status == Status::Unbounded) ||
                     (!primal && result.status == Status::Infeasible))
                value = std::numeric_limits<double>::infinity();
            else
                throw std::runtime_error("Unexpected feasibility status for Roth LP.");
            best = std::max(best, value);
            // Only optimal inner dual values certify the outer maximum;
            // never stop using an unfinished minimization's primal objective.
            if (best > threshold || std::isinf(best)) {
                stop = true;
                break;
            }
        }
    };
    enumerate(0, m);
    return std::min(best, threshold);
}
} // namespace

double h_m_roth_primal_lp_glpk(const Eigen::MatrixXd& G, int m, double early_quit_threshold) {
    return roth_lp(G, m, early_quit_threshold, Backend::Glpk, true);
}
double h_m_roth_dual_lp_glpk(const Eigen::MatrixXd& G, int m, double early_quit_threshold) {
    return roth_lp(G, m, early_quit_threshold, Backend::Glpk, false);
}

#ifdef HAVE_HIGHS
namespace {
double roth_highs(const LpInput& G, int m, double threshold, int num_threads, bool primal) {
    using namespace m_height_lp;
    validate_highs_input(G, m, num_threads, threshold);
    if (m == 0) return std::min(1.0, threshold);
    const int k = G.rows(), n = G.cols(), q = n - m;
    struct Job { std::vector<int> S, complement; };
    Combinations combinations(n, m);
    bool more = true;
    auto next = [&](Job& job) {
        if (!more) return false;
        job.S = combinations.values;
        job.complement.clear();
        for (int j = 0; j < n; ++j)
            if (!std::binary_search(job.S.begin(), job.S.end(), j)) job.complement.push_back(j);
        more = combinations.advance();
        return true;
    };
    auto evaluate = [&](HighsWorkspace& w, const Job& job, const auto& cancelled) {
        if (primal) {
            std::fill(w.lp.row_upper_.begin(), w.lp.row_upper_.end(), 1.0);
        } else {
            std::fill(w.lp.col_cost_.begin(), w.lp.col_cost_.end(), 1.0);
            std::fill(w.lp.col_lower_.begin(), w.lp.col_lower_.end(), 0.0);
        }
        for (int t = 0; t < q; ++t)
            for (int c = 0; c < k; ++c) {
                const double g = G(c, job.complement[t]);
                if (primal) {
                    w.coefficient(2 * t, c) = g;
                    w.coefficient(2 * t + 1, c) = -g;
                } else {
                    w.coefficient(c, t) = g;
                    w.coefficient(c, q + t) = -g;
                }
            }
        double best = -std::numeric_limits<double>::infinity();
        for (int i : job.S) {
            if (cancelled()) break;
            for (int c = 0; c < k; ++c) {
                if (primal) w.lp.col_cost_[c] = G(c, i);
                else w.lp.row_lower_[c] = w.lp.row_upper_[c] = G(c, i);
            }
            const auto result = w.solve();
            double value;
            if (primal) value = primal_value(result);
            else if (result.status == Status::Optimal) value = result.value;
            else if (result.status == Status::Infeasible) value = std::numeric_limits<double>::infinity();
            else throw std::runtime_error("Roth dual LP unexpectedly unbounded.");
            best = std::max(best, value);
            if (best > threshold || best == std::numeric_limits<double>::infinity()) break;
        }
        return best;
    };
    return highs_sweep<Job>(primal ? 2 * q : k, primal ? k : 2 * q, primal,
                            num_threads, threshold, next, evaluate);
}
} // namespace
double h_m_roth_primal_lp_highs(const LpInput& G, int m, double early_quit_threshold, int num_threads) {
    return roth_highs(G, m, early_quit_threshold, num_threads, true);
}
double h_m_roth_dual_lp_highs(const LpInput& G, int m, double early_quit_threshold, int num_threads) {
    return roth_highs(G, m, early_quit_threshold, num_threads, false);
}
#endif
