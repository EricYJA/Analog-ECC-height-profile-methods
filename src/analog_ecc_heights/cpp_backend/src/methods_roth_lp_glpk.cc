#include "methods.hh"
#include "lp_workspace_glpk.hh"

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
    // Use exact zero detection: any nonzero row space has h_0 = 1.
    if (m == 0) return std::min((G.array() == 0.0).all() ? 0.0 : 1.0, threshold);

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

