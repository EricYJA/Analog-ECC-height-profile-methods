#include "methods.hh"
#include "utils.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace comb = m_height_utils::detail;
constexpr double inf = std::numeric_limits<double>::infinity();

template <typename Fn>
void for_each_sign_vector(int k, Fn&& fn) {
    if (k < 0) return;
    std::vector<double> signs(k, -1.0);
    if (k == 0) {
        fn(signs);
        return;
    }
    signs[0] = 1.0;

    std::function<void(int)> dfs = [&](int idx) {
        if (idx == k) {
            fn(signs);
            return;
        }
        signs[idx] = -1.0;
        dfs(idx + 1);
        signs[idx] = 1.0;
        dfs(idx + 1);
    };
    dfs(1);
}

struct ComplementCandidateStats {
    double infinity_norm = 0.0;
    int greater_than_one = 0;
    int less_than_one = 0;
};

ComplementCandidateStats complement_candidate_stats(
    const Eigen::VectorXd& complement_values)
{
    ComplementCandidateStats stats;
    for (int idx = 0; idx < complement_values.size(); ++idx) {
        const double magnitude = std::abs(complement_values[idx]);
        stats.infinity_norm = std::max(stats.infinity_norm, magnitude);
        if (magnitude > 1.0) {
            ++stats.greater_than_one;
        } else if (magnitude < 1.0) {
            ++stats.less_than_one;
        }
    }
    return stats;
}

double roth_primal_complement_candidate_value(
    const Eigen::VectorXd& complement_values,
    int m,
    int n)
{
    const ComplementCandidateStats stats =
        complement_candidate_stats(complement_values);

    // Roth Theorem 5, Eq. (15): u is admissible only when the
    // complement has at most m entries above 1 and at most n-m-1 below 1.
    if (stats.greater_than_one > m || stats.less_than_one > n - m - 1) {
        return -std::numeric_limits<double>::infinity();
    }
    return stats.infinity_norm;
}

void update_roth_primal_complement_all_max(
    const Eigen::VectorXd& complement_values,
    int n,
    std::vector<double>& best)
{
    const int max_m = static_cast<int>(best.size());
    if (max_m == 0) return;

    const ComplementCandidateStats stats =
        complement_candidate_stats(complement_values);
    for (int m = 1; m <= max_m; ++m) {
        if (stats.greater_than_one <= m && stats.less_than_one <= n - m - 1) {
            best[m - 1] = std::max(best[m - 1], stats.infinity_norm);
        }
    }
}

// R u = G_{I^c}^T G_I^{-T} u. A single multiple-RHS solve
// supplies the same transform to scalar, full-profile, and pruning paths.
bool build_primal_transform(const Eigen::MatrixXd& G, const std::vector<int>& I,
                            double tol, double scale, Eigen::MatrixXd& R) {
    const auto lu = comb::factor_columns(G, I, tol, scale);
    if (lu.rank() != G.rows()) return false;
    const auto Ic = comb::complement_indices(static_cast<int>(G.cols()), I);
    R = lu.solve(comb::gather_columns(G, Ic)).transpose();
    return true;
}

// Reorder sign coordinates so that the most influential columns of R are branched first.
// Here we use descending l1 norm as a simple exact heuristic.
void reorder_R_columns_by_l1(Eigen::MatrixXd& R) {
    const int k = static_cast<int>(R.cols());
    std::vector<int> order(k);
    for (int t = 0; t < k; ++t) order[t] = t;

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return R.col(a).cwiseAbs().sum() > R.col(b).cwiseAbs().sum();
    });

    Eigen::MatrixXd R_perm(R.rows(), R.cols());
    for (int t = 0; t < k; ++t) R_perm.col(t) = R.col(order[t]);
    R.swap(R_perm);
}

// suffix_abs.col(d) = sum_{t=d}^{k-1} |R.col(t)| on the complement coordinates.
// This gives the remaining coordinate-wise uncertainty after fixing the first d signs.
Eigen::MatrixXd build_suffix_abs(const Eigen::MatrixXd& R) {
    const int n = static_cast<int>(R.rows());
    const int k = static_cast<int>(R.cols());

    Eigen::MatrixXd suffix_abs(n, k + 1);
    suffix_abs.col(k).setZero();
    for (int t = k - 1; t >= 0; --t) {
        suffix_abs.col(t) = suffix_abs.col(t + 1) + R.col(t).cwiseAbs();
    }
    return suffix_abs;
}

// Depth-first search over sign bits with exact infinity-norm pruning.
// complement_partial stores sum_{t<depth} u_t R.col(t) on bar I.
void dfs_signs_branch_and_bound(
    int depth,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& suffix_abs,
    Eigen::VectorXd& complement_partial,
    int m,
    int n,
    double& best_for_I)
{
    const int k = static_cast<int>(R.cols());

    // Every completion is bounded coordinate-wise by the remaining absolute
    // contribution. This directly bounds the Eq. (15) infinity norm.
    const double upper_bound = complement_partial.size() == 0
        ? 0.0
        : (complement_partial.cwiseAbs() + suffix_abs.col(depth)).maxCoeff();
    if (upper_bound <= best_for_I) {
        return;
    }

    if (depth == k) {
        best_for_I = std::max(
            best_for_I,
            roth_primal_complement_candidate_value(complement_partial, m, n));
        return;
    }

    // Branch u_depth = +1 first.
    complement_partial += R.col(depth);
    dfs_signs_branch_and_bound(
        depth + 1, R, suffix_abs, complement_partial, m, n, best_for_I);

    // Then u_depth = -1.
    complement_partial -= 2.0 * R.col(depth);
    dfs_signs_branch_and_bound(
        depth + 1, R, suffix_abs, complement_partial, m, n, best_for_I);

    // Restore.
    complement_partial += R.col(depth);
}

}  // namespace

double h_m_roth_primal_combinatorial(
    const Eigen::MatrixXd& G, int m, double tol, int num_threads) {
    m_height_utils::validate_num_threads(num_threads);
    m_height_utils::validate_generator_matrix(G, tol);
    const int k = static_cast<int>(G.rows()), n = static_cast<int>(G.cols());
    m_height_utils::validate_m_for_columns(m, n);
    if (m == 0) return 1.0;
    if (!comb::generator_minimum_distance_exceeds_validated(G, m, tol)) return inf;

    const auto all_I = comb::collect_subsets(n, k);
    const double scale = comb::matrix_scale(G);
    double best = -inf;
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads) schedule(dynamic) reduction(max: best)
    for (int idx = 0; idx < static_cast<int>(all_I.size()); ++idx) {
        Eigen::MatrixXd R;
        if (!build_primal_transform(G, all_I[idx], tol, scale, R)) continue;
        Eigen::VectorXd values(n - k);
        for_each_sign_vector(k, [&](const auto& signs) {
            const Eigen::Map<const Eigen::VectorXd> u(signs.data(), k);
            values.noalias() = R * u;
            best = std::max(best, roth_primal_complement_candidate_value(values, m, n));
        });
    }
    return best;
}

std::vector<double> h_m_roth_primal_combinatorial(
    const Eigen::MatrixXd& G, std::nullopt_t, double tol, int num_threads) {
    m_height_utils::validate_num_threads(num_threads);
    m_height_utils::validate_generator_matrix(G, tol);
    const int k = static_cast<int>(G.rows()), n = static_cast<int>(G.cols());
    const int max_m = n - k;
    if (max_m == 0) return {};
    const int finite_m_count = std::min(
        max_m, comb::generator_minimum_distance_validated(G, tol) - 1);
    std::vector<double> result(max_m, inf);
    if (finite_m_count == 0) return result;

    const auto all_I = comb::collect_subsets(n, k);
    const double scale = comb::matrix_scale(G);
    std::fill_n(result.begin(), finite_m_count, -inf);
#pragma omp parallel if(num_threads > 1) num_threads(num_threads)
    {
        std::vector<double> best_private(finite_m_count, -inf);
#pragma omp for schedule(dynamic) nowait
        for (int idx = 0; idx < static_cast<int>(all_I.size()); ++idx) {
            Eigen::MatrixXd R;
            if (!build_primal_transform(G, all_I[idx], tol, scale, R)) continue;
            Eigen::VectorXd values(n - k);
            for_each_sign_vector(k, [&](const auto& signs) {
                const Eigen::Map<const Eigen::VectorXd> u(signs.data(), k);
                values.noalias() = R * u;
                update_roth_primal_complement_all_max(values, n, best_private);
            });
        }
#pragma omp critical
        {
            for (int i = 0; i < finite_m_count; ++i)
                result[i] = std::max(result[i], best_private[i]);
        }
    }
    return result;
}

double h_m_roth_primal_combinatorial_pruning(
    const Eigen::MatrixXd& G, int m, double tol, int num_threads) {
    m_height_utils::validate_num_threads(num_threads);
    m_height_utils::validate_generator_matrix(G, tol);
    const int k = static_cast<int>(G.rows()), n = static_cast<int>(G.cols());
    m_height_utils::validate_m_for_columns(m, n);
    if (m == 0) return 1.0;
    if (!comb::generator_minimum_distance_exceeds_validated(G, m, tol)) return inf;

    const auto all_I = comb::collect_subsets(n, k);
    const double scale = comb::matrix_scale(G);
    double best = -inf;
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads) schedule(dynamic) reduction(max: best)
    for (int idx = 0; idx < static_cast<int>(all_I.size()); ++idx) {
        Eigen::MatrixXd R;
        if (!build_primal_transform(G, all_I[idx], tol, scale, R)) continue;
        reorder_R_columns_by_l1(R);
        const Eigen::MatrixXd suffix_abs = build_suffix_abs(R);
        // Global sign symmetry permits fixing the first (reordered) sign.
        Eigen::VectorXd partial = R.col(0);
        double best_for_I = -inf;
        dfs_signs_branch_and_bound(1, R, suffix_abs, partial, m, n, best_for_I);
        best = std::max(best, best_for_I);
    }
    return best;
}

double h_m_roth_dual_combinatorial_generator(
    const Eigen::MatrixXd& G, int m, double tol, int num_threads) {
    m_height_utils::validate_num_threads(num_threads);
    m_height_utils::validate_generator_matrix(G, tol);
    const int k = static_cast<int>(G.rows()), n = static_cast<int>(G.cols());
    m_height_utils::validate_m_for_columns(m, n);
    if (m == 0) return 1.0;
    if (!comb::generator_minimum_distance_exceeds_validated(G, m, tol)) return inf;

    const auto all_S = comb::collect_subsets(n, m);
    const double scale = comb::matrix_scale(G);
    double best = -inf;
    int missing_basis = 0;
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads) schedule(dynamic) reduction(max: best) reduction(|: missing_basis)
    for (int idx = 0; idx < static_cast<int>(all_S.size()); ++idx) {
        const auto& S = all_S[idx];
        const auto Sc = comb::complement_indices(n, S);
        const Eigen::MatrixXd targets = comb::gather_columns(G, S);
        Eigen::MatrixXd coefficients(k, m);
        Eigen::RowVectorXd minima = Eigen::RowVectorXd::Constant(m, inf);
        bool found = false;
        comb::for_each_subset_from_pool(Sc, k, [&](const auto& I) {
            const auto lu = comb::factor_columns(G, I, tol, scale);
            if (lu.rank() != k) return;
            coefficients = lu.solve(targets);
            minima = minima.cwiseMin(coefficients.cwiseAbs().colwise().sum());
            found = true;
        });
        // Keep a separate minimum for each i; max_i min_I is NOT min_I max_i.
        if (found) best = std::max(best, minima.maxCoeff());
        else missing_basis = 1;
    }
    if (missing_basis) throw std::runtime_error("No invertible I subset of S^c was found.");
    return best;
}

double h_m_roth_dual_combinatorial_parity(
    const Eigen::MatrixXd& H, int m, double tol, int num_threads) {
    m_height_utils::validate_num_threads(num_threads);
    m_height_utils::validate_parity_check_matrix(H, tol);
    const int r = static_cast<int>(H.rows()), n = static_cast<int>(H.cols());
    m_height_utils::validate_m_for_columns(m, n);
    if (m == 0) return 1.0;
    if (!comb::parity_check_minimum_distance_exceeds_validated(H, m, tol)) return inf;

    const auto all_S = comb::collect_subsets(n, m);
    const double scale = comb::matrix_scale(H);
    double best = -inf;
    int missing_basis = 0;
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads) schedule(dynamic) reduction(max: best) reduction(|: missing_basis)
    for (int idx = 0; idx < static_cast<int>(all_S.size()); ++idx) {
        const auto& S = all_S[idx];
        const auto Sc = comb::complement_indices(n, S);
        Eigen::VectorXd minima = Eigen::VectorXd::Constant(m, inf);
        Eigen::MatrixXd coefficients(r, n - r);
        std::vector<int> J(r);
        bool found = false;
        // Generate only supersets J of S, without a global invertibility cache.
        comb::for_each_subset_from_pool(Sc, r - m, [&](const auto& extra) {
            std::merge(S.begin(), S.end(), extra.begin(), extra.end(), J.begin());
            const auto lu = comb::factor_columns(H, J, tol, scale);
            if (lu.rank() != r) return;
            coefficients = lu.solve(comb::gather_columns(H, comb::complement_indices(n, J)));
            for (int t = 0; t < m; ++t) {
                const int row = static_cast<int>(
                    std::lower_bound(J.begin(), J.end(), S[t]) - J.begin());
                minima[t] = std::min(minima[t], coefficients.row(row).cwiseAbs().sum());
            }
            found = true;
        });
        if (found) best = std::max(best, minima.maxCoeff());
        else missing_basis = 1;
    }
    if (missing_basis) throw std::runtime_error("No invertible J superset of S was found.");
    return best;
}
