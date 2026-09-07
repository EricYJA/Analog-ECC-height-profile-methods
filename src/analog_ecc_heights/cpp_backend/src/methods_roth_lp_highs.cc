#include "methods.hh"
#include "lp_workspace_highs.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
double roth_highs(const LpInput& G, int m, double threshold, int num_threads, bool primal) {
    using namespace m_height_lp;
    validate_highs_input(G, m, num_threads, threshold);
    // Use exact zero detection: any nonzero row space has h_0 = 1.
    if (m == 0) return std::min((G.array() == 0.0).all() ? 0.0 : 1.0, threshold);
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
