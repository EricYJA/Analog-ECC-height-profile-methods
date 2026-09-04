#include "methods.hh"
#include "utils.hh"

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void validate_m_for_mds_case(int m, int n, int r) {
    if (m < 0 || m > n - 1) {
        std::ostringstream oss;
        oss << "m must satisfy 0 <= m <= n-1, got m=" << m << ", n=" << n;
        throw std::invalid_argument(oss.str());
    }
    if (m != r) {
        std::ostringstream oss;
        oss << "MDS combinatorial form requires m = n-k = " << r << ", got m=" << m;
        throw std::invalid_argument(oss.str());
    }
}

template <typename Fn>
void for_each_subset(int n, int subset_size, Fn&& fn) {
    if (subset_size < 0 || subset_size > n) return;

    std::vector<int> subset;
    subset.reserve(subset_size);

    std::function<void(int, int)> dfs = [&](int start, int need) {
        if (need == 0) {
            fn(subset);
            return;
        }
        for (int value = start; value <= n - need; ++value) {
            subset.push_back(value);
            dfs(value + 1, need - 1);
            subset.pop_back();
        }
    };
    dfs(0, subset_size);
}

std::vector<std::vector<int>> collect_subsets(int n, int subset_size) {
    std::vector<std::vector<int>> subsets;
    for_each_subset(n, subset_size, [&](const std::vector<int>& subset) {
        subsets.push_back(subset);
    });
    return subsets;
}

std::vector<int> complement_indices(int n, const std::vector<int>& subset) {
    std::vector<char> is_in_subset(n, 0);
    for (int idx : subset) is_in_subset[idx] = 1;

    std::vector<int> out;
    out.reserve(n - static_cast<int>(subset.size()));
    for (int idx = 0; idx < n; ++idx) {
        if (!is_in_subset[idx]) out.push_back(idx);
    }
    return out;
}

Eigen::MatrixXd gather_columns(const Eigen::MatrixXd& G, const std::vector<int>& cols) {
    Eigen::MatrixXd out(G.rows(), static_cast<int>(cols.size()));
    for (int j = 0; j < static_cast<int>(cols.size()); ++j) out.col(j) = G.col(cols[j]);
    return out;
}

}  // namespace

double h_m_roth_mds_combinatorial(const Eigen::MatrixXd& G, int m, double tol) {
    m_height_utils::validate_generator_matrix(G, tol);

    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    const int r = n - k;
    validate_m_for_mds_case(m, n, r);

    if (r == 0) return 1.0;
    if (!m_height_utils::generator_minimum_distance_exceeds(G, m, tol)) {
        throw std::invalid_argument(
            "G must generate an MDS code: every k-column submatrix must be nonsingular.");
    }

    const double reference_scale = m_height_utils::detail::matrix_scale(G);
    double best = -std::numeric_limits<double>::infinity();
    for_each_subset(n, r, [&](const std::vector<int>& S) {
        const std::vector<int> Sc = complement_indices(n, S);
        const Eigen::MatrixXd GSc = gather_columns(G, Sc);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(GSc);
        m_height_utils::detail::set_numerical_rank_threshold(lu, tol, reference_scale);
        for (int i : S) {
            const Eigen::VectorXd coeffs = lu.solve(G.col(i));
            best = std::max(best, coeffs.lpNorm<1>());
        }
    });

    return best;
}

double h_m_roth_mds_combinatorial_omp(const Eigen::MatrixXd& G, int m, double tol) {
    m_height_utils::validate_generator_matrix(G, tol);

    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    const int r = n - k;
    validate_m_for_mds_case(m, n, r);

    if (r == 0) return 1.0;

#ifndef _OPENMP
    return h_m_roth_mds_combinatorial(G, m, tol);
#else
    if (!m_height_utils::generator_minimum_distance_exceeds(G, m, tol)) {
        throw std::invalid_argument(
            "G must generate an MDS code: every k-column submatrix must be nonsingular.");
    }
    const std::vector<std::vector<int>> all_S = collect_subsets(n, r);
    const double reference_scale = m_height_utils::detail::matrix_scale(G);

    double best = -std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(dynamic) reduction(max: best)
    for (int idx = 0; idx < static_cast<int>(all_S.size()); ++idx) {
        const std::vector<int>& S = all_S[idx];
        const std::vector<int> Sc = complement_indices(n, S);
        const Eigen::MatrixXd GSc = gather_columns(G, Sc);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(GSc);
        m_height_utils::detail::set_numerical_rank_threshold(lu, tol, reference_scale);

        double best_for_S = -std::numeric_limits<double>::infinity();
        for (int i : S) {
            const Eigen::VectorXd coeffs = lu.solve(G.col(i));
            best_for_S = std::max(best_for_S, coeffs.lpNorm<1>());
        }
        best = std::max(best, best_for_S);
    }

    return best;
#endif
}
