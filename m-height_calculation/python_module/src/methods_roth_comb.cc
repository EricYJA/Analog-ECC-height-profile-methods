#include "methods.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kHmTol = 1e-12;

void validate_nonempty_matrix(const Eigen::MatrixXd& M, const char* name) {
    if (M.rows() == 0 || M.cols() == 0) {
        throw std::runtime_error(std::string(name) + " must be a non-empty 2D matrix.");
    }
}

void validate_m_for_columns(int m, int n) {
    if (m < 0 || m > n - 1) {
        std::ostringstream oss;
        oss << "m must satisfy 0 <= m <= n-1, got m=" << m << ", n=" << n;
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
        for (int v = start; v <= n - need; ++v) {
            subset.push_back(v);
            dfs(v + 1, need - 1);
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

template <typename Fn>
void for_each_subset_from_pool(const std::vector<int>& pool, int subset_size, Fn&& fn) {
    const int n = static_cast<int>(pool.size());
    if (subset_size < 0 || subset_size > n) return;

    std::vector<int> subset;
    subset.reserve(subset_size);

    std::function<void(int, int)> dfs = [&](int start, int need) {
        if (need == 0) {
            fn(subset);
            return;
        }
        for (int idx = start; idx <= n - need; ++idx) {
            subset.push_back(pool[idx]);
            dfs(idx + 1, need - 1);
            subset.pop_back();
        }
    };
    dfs(0, subset_size);
}

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

std::vector<int> complement_indices(int n, const std::vector<int>& subset) {
    std::vector<char> in_subset(n, 0);
    for (int idx : subset) {
        if (idx < 0 || idx >= n) throw std::out_of_range("subset index out of range");
        in_subset[idx] = 1;
    }
    std::vector<int> out;
    out.reserve(n - static_cast<int>(subset.size()));
    for (int i = 0; i < n; ++i) {
        if (!in_subset[i]) out.push_back(i);
    }
    return out;
}

Eigen::MatrixXd gather_columns(const Eigen::MatrixXd& M, const std::vector<int>& cols) {
    Eigen::MatrixXd out(M.rows(), static_cast<int>(cols.size()));
    for (int j = 0; j < static_cast<int>(cols.size()); ++j) out.col(j) = M.col(cols[j]);
    return out;
}

bool is_nonsingular(const Eigen::MatrixXd& M, double tol) {
    if (M.rows() == 0 || M.rows() != M.cols()) return false;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    lu.setThreshold(tol);
    return lu.rank() == M.rows();
}

struct ComplementCandidateStats {
    double infinity_norm = 0.0;
    int greater_than_one = 0;
    int less_than_one = 0;
};

ComplementCandidateStats complement_candidate_stats(
    const Eigen::VectorXd& complement_values,
    double unit_tol = kHmTol)
{
    ComplementCandidateStats stats;
    for (int idx = 0; idx < complement_values.size(); ++idx) {
        const double magnitude = std::abs(complement_values[idx]);
        stats.infinity_norm = std::max(stats.infinity_norm, magnitude);
        if (magnitude > 1.0 + unit_tol) {
            ++stats.greater_than_one;
        } else if (magnitude < 1.0 - unit_tol) {
            ++stats.less_than_one;
        }
    }
    return stats;
}

double roth_primal_complement_candidate_value(
    const Eigen::VectorXd& complement_values,
    int m,
    int n,
    double unit_tol = kHmTol)
{
    validate_m_for_columns(m, n);
    const ComplementCandidateStats stats =
        complement_candidate_stats(complement_values, unit_tol);

    // Roth Theorem 5, Eq. (15): u is admissible only when the
    // complement has at most m entries above 1 and at most n-m-1 below 1.
    if (stats.greater_than_one > m || stats.less_than_one > n - m - 1) {
        return -std::numeric_limits<double>::infinity();
    }
    return stats.infinity_norm;
}

int validate_and_get_all_m_count_for_generator(const Eigen::MatrixXd& G) {
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    if (n < k) {
        std::ostringstream oss;
        oss << "G must satisfy rows <= cols, got rows=" << k << ", cols=" << n;
        throw std::invalid_argument(oss.str());
    }
    return n - k;
}

void update_roth_primal_complement_all_max(
    const Eigen::VectorXd& complement_values,
    int n,
    std::vector<double>& best,
    double unit_tol = kHmTol)
{
    const int max_m = static_cast<int>(best.size());
    if (max_m == 0) return;
    if (max_m > n - 1) {
        std::ostringstream oss;
        oss << "max m must satisfy 0 <= max_m <= n-1, got max_m=" << max_m << ", n=" << n;
        throw std::invalid_argument(oss.str());
    }

    const ComplementCandidateStats stats =
        complement_candidate_stats(complement_values, unit_tol);
    for (int m = 1; m <= max_m; ++m) {
        if (stats.greater_than_one <= m && stats.less_than_one <= n - m - 1) {
            best[m - 1] = std::max(best[m - 1], stats.infinity_norm);
        }
    }
}

void merge_max_values(std::vector<double>& dst, const std::vector<double>& src) {
    if (dst.size() != src.size()) {
        throw std::invalid_argument("merge_max_values requires vectors of the same size.");
    }
    for (int idx = 0; idx < static_cast<int>(dst.size()); ++idx) {
        dst[idx] = std::max(dst[idx], src[idx]);
    }
}

std::string subset_key(const std::vector<int>& subset) {
    std::ostringstream oss;
    for (int i = 0; i < static_cast<int>(subset.size()); ++i) {
        if (i) oss << ",";
        oss << subset[i];
    }
    return oss.str();
}

int position_in_sorted_subset(const std::vector<int>& subset, int value) {
    auto it = std::lower_bound(subset.begin(), subset.end(), value);
    if (it == subset.end() || *it != value) return -1;
    return static_cast<int>(it - subset.begin());
}


// Build R = M^T * (G_I^T)^{-1} from an existing LU factorization of G_I^T.
// For M = G_{bar I}, the complement vector in Roth Eq. (15) is R u.
Eigen::MatrixXd build_R_from_lu_and_G(
    const Eigen::PartialPivLU<Eigen::MatrixXd>& lu,
    const Eigen::MatrixXd& M)
{
    const int k = static_cast<int>(M.rows());
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(k, k);
    const Eigen::MatrixXd inv_GIt = lu.solve(I);   // (G_I^T)^{-1}
    return M.transpose() * inv_GIt;
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

#ifdef _OPENMP
std::vector<std::vector<int>> collect_nonsingular_subsets_parallel(
    const Eigen::MatrixXd& M,
    int subset_size,
    double tol)
{
    const auto subsets = collect_subsets(static_cast<int>(M.cols()), subset_size);
    std::vector<char> keep(subsets.size(), 0);

#pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < static_cast<int>(subsets.size()); ++idx) {
        keep[idx] = is_nonsingular(gather_columns(M, subsets[idx]), tol) ? 1 : 0;
    }

    std::vector<std::vector<int>> good_subsets;
    good_subsets.reserve(subsets.size());
    for (int idx = 0; idx < static_cast<int>(subsets.size()); ++idx) {
        if (keep[idx]) good_subsets.push_back(subsets[idx]);
    }
    return good_subsets;
}

std::unordered_set<std::string> collect_nonsingular_subset_keys_parallel(
    const Eigen::MatrixXd& M,
    int subset_size,
    double tol)
{
    const auto subsets = collect_nonsingular_subsets_parallel(M, subset_size, tol);
    std::unordered_set<std::string> good_subset_keys;
    good_subset_keys.reserve(subsets.size());
    for (const auto& subset : subsets) good_subset_keys.insert(subset_key(subset));
    return good_subset_keys;
}
#endif

}  // namespace

double h_m_roth_primal_combinatorial(const Eigen::MatrixXd& G, int m, double tol) {
    validate_nonempty_matrix(G, "G");
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    validate_m_for_columns(m, n);
    if (m == 0) return 1.0;

    std::vector<std::vector<int>> good_I;
    for_each_subset(n, k, [&](const std::vector<int>& I) {
        if (is_nonsingular(gather_columns(G, I), tol)) good_I.push_back(I);
    });

    double best = -std::numeric_limits<double>::infinity();
    for (const auto& I : good_I) {
        const Eigen::MatrixXd GI = gather_columns(G, I);
        const Eigen::MatrixXd GIc = gather_columns(G, complement_indices(n, I));
        const Eigen::PartialPivLU<Eigen::MatrixXd> lu(GI.transpose());

        for_each_sign_vector(k, [&](const std::vector<double>& signs) {
            const Eigen::Map<const Eigen::VectorXd> u(signs.data(), k);
            const Eigen::VectorXd alpha = lu.solve(u);
            const Eigen::VectorXd complement_values =
                (alpha.transpose() * GIc).transpose();
            best = std::max(
                best,
                roth_primal_complement_candidate_value(complement_values, m, n));
        });
    }
    return best;
}

double h_m_roth_primal_combinatorial_omp(const Eigen::MatrixXd& G, int m, double tol) {
    validate_nonempty_matrix(G, "G");
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    validate_m_for_columns(m, n);
    if (m == 0) return 1.0;

#ifndef _OPENMP
    return h_m_roth_primal_combinatorial(G, m, tol);
#else
    const std::vector<std::vector<int>> good_I = collect_nonsingular_subsets_parallel(G, k, tol);

    double best = -std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(dynamic) reduction(max: best)
    for (int idx = 0; idx < static_cast<int>(good_I.size()); ++idx) {
        double best_for_I = -std::numeric_limits<double>::infinity();
        const Eigen::MatrixXd GI = gather_columns(G, good_I[idx]);
        const Eigen::MatrixXd GIc =
            gather_columns(G, complement_indices(n, good_I[idx]));
        const Eigen::PartialPivLU<Eigen::MatrixXd> lu(GI.transpose());

        for_each_sign_vector(k, [&](const std::vector<double>& signs) {
            const Eigen::Map<const Eigen::VectorXd> u(signs.data(), k);
            const Eigen::VectorXd alpha = lu.solve(u);
            const Eigen::VectorXd complement_values =
                (alpha.transpose() * GIc).transpose();
            best_for_I = std::max(
                best_for_I,
                roth_primal_complement_candidate_value(complement_values, m, n));
        });

        best = std::max(best, best_for_I);
    }
    return best;
#endif
}

std::vector<double> h_m_roth_primal_combinatorial_omp_all(const Eigen::MatrixXd& G, double tol) {
    validate_nonempty_matrix(G, "G");
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    const int max_m = validate_and_get_all_m_count_for_generator(G);
    if (max_m == 0) return {};

#ifndef _OPENMP
    std::vector<std::vector<int>> good_I;
    for_each_subset(static_cast<int>(G.cols()), k, [&](const std::vector<int>& I) {
        if (is_nonsingular(gather_columns(G, I), tol)) good_I.push_back(I);
    });

    std::vector<double> best(max_m, -std::numeric_limits<double>::infinity());
    for (const auto& I : good_I) {
        std::vector<double> best_for_I(max_m, -std::numeric_limits<double>::infinity());
        const Eigen::MatrixXd GI = gather_columns(G, I);
        const Eigen::MatrixXd GIc = gather_columns(G, complement_indices(n, I));
        const Eigen::PartialPivLU<Eigen::MatrixXd> lu(GI.transpose());

        for_each_sign_vector(k, [&](const std::vector<double>& signs) {
            const Eigen::Map<const Eigen::VectorXd> u(signs.data(), k);
            const Eigen::VectorXd alpha = lu.solve(u);
            const Eigen::VectorXd complement_values =
                (alpha.transpose() * GIc).transpose();
            update_roth_primal_complement_all_max(
                complement_values, n, best_for_I);
        });

        merge_max_values(best, best_for_I);
    }
    return best;
#else
    const std::vector<std::vector<int>> good_I = collect_nonsingular_subsets_parallel(G, k, tol);
    std::vector<double> best(max_m, -std::numeric_limits<double>::infinity());

#pragma omp parallel
    {
        std::vector<double> best_private(max_m, -std::numeric_limits<double>::infinity());

#pragma omp for schedule(dynamic) nowait
        for (int idx = 0; idx < static_cast<int>(good_I.size()); ++idx) {
            std::vector<double> best_for_I(max_m, -std::numeric_limits<double>::infinity());
            const Eigen::MatrixXd GI = gather_columns(G, good_I[idx]);
            const Eigen::MatrixXd GIc =
                gather_columns(G, complement_indices(n, good_I[idx]));
            const Eigen::PartialPivLU<Eigen::MatrixXd> lu(GI.transpose());

            for_each_sign_vector(k, [&](const std::vector<double>& signs) {
                const Eigen::Map<const Eigen::VectorXd> u(signs.data(), k);
                const Eigen::VectorXd alpha = lu.solve(u);
                const Eigen::VectorXd complement_values =
                    (alpha.transpose() * GIc).transpose();
                update_roth_primal_complement_all_max(
                    complement_values, n, best_for_I);
            });

            merge_max_values(best_private, best_for_I);
        }

#pragma omp critical
        merge_max_values(best, best_private);
    }

    return best;
#endif
}


double h_m_roth_primal_combinatorial_pruning_omp(const Eigen::MatrixXd& G, int m, double tol) {
    validate_nonempty_matrix(G, "G");
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    validate_m_for_columns(m, n);
    if (m == 0) return 1.0;

#ifndef _OPENMP
    // This replacement does not actually require OpenMP for correctness,
    // but keep the fallback style consistent with the current file.
    return h_m_roth_primal_combinatorial(G, m, tol);
#else
    const std::vector<std::vector<int>> good_I = collect_nonsingular_subsets_parallel(G, k, tol);

    double best = -std::numeric_limits<double>::infinity();

#pragma omp parallel for schedule(dynamic) reduction(max: best)
    for (int idx = 0; idx < static_cast<int>(good_I.size()); ++idx) {
        double best_for_I = -std::numeric_limits<double>::infinity();

        const Eigen::MatrixXd GI = gather_columns(G, good_I[idx]);
        const Eigen::MatrixXd GIc =
            gather_columns(G, complement_indices(n, good_I[idx]));
        const Eigen::PartialPivLU<Eigen::MatrixXd> lu(GI.transpose());

        // Build the complement values y(u) = R u once for this subset I.
        Eigen::MatrixXd R = build_R_from_lu_and_G(lu, GIc);

        // Reorder sign coordinates to improve pruning.
        reorder_R_columns_by_l1(R);

        // Coordinate-wise remaining uncertainty after each depth.
        const Eigen::MatrixXd suffix_abs = build_suffix_abs(R);

        // Global sign symmetry preserves both the infinity norm and the
        // Eq. (15) admissibility counts.
        // So fix the first sign to +1 and search only half of the hypercube.
        Eigen::VectorXd complement_partial = R.col(0);  // depth = 1, u_0 = +1

        if (k == 1) {
            // Only one representative leaf remains after symmetry fixing.
            best_for_I = std::max(
                best_for_I,
                roth_primal_complement_candidate_value(
                    complement_partial, m, n));
        } else {
            dfs_signs_branch_and_bound(
                /*depth=*/1,
                R,
                suffix_abs,
                complement_partial,
                m,
                n,
                best_for_I);
        }

        best = std::max(best, best_for_I);
    }

    return best;
#endif
}

double h_m_roth_dual_combinatorial_generator(const Eigen::MatrixXd& G, int m, double tol) {
    validate_nonempty_matrix(G, "G");
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    validate_m_for_columns(m, n);

    std::unordered_set<std::string> invertible;
    for_each_subset(n, k, [&](const std::vector<int>& I) {
        if (is_nonsingular(gather_columns(G, I), tol)) invertible.insert(subset_key(I));
    });

    double best_outer = -std::numeric_limits<double>::infinity();
    for_each_subset(n, m, [&](const std::vector<int>& S) {
        const std::vector<int> Sc = complement_indices(n, S);
        for (int i : S) {
            double inner_best = std::numeric_limits<double>::infinity();
            bool found = false;

            for_each_subset_from_pool(Sc, k, [&](const std::vector<int>& I) {
                if (!invertible.count(subset_key(I))) return;
                const Eigen::MatrixXd GI = gather_columns(G, I);
                const Eigen::VectorXd coeffs = GI.colPivHouseholderQr().solve(G.col(i));
                inner_best = std::min(inner_best, coeffs.lpNorm<1>());
                found = true;
            });

            if (!found) {
                throw std::runtime_error("No invertible I subset of S^c was found.");
            }
            best_outer = std::max(best_outer, inner_best);
        }
    });
    return best_outer;
}

double h_m_roth_dual_combinatorial_generator_omp(const Eigen::MatrixXd& G, int m, double tol) {
    validate_nonempty_matrix(G, "G");
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    validate_m_for_columns(m, n);

#ifndef _OPENMP
    return h_m_roth_dual_combinatorial_generator(G, m, tol);
#else
    const std::unordered_set<std::string> invertible =
        collect_nonsingular_subset_keys_parallel(G, k, tol);
    const std::vector<std::vector<int>> all_S = collect_subsets(n, m);

    double best_outer = -std::numeric_limits<double>::infinity();
    int missing_invertible_subset = 0;
#pragma omp parallel for schedule(dynamic) reduction(max: best_outer) reduction(|: missing_invertible_subset)
    for (int s_idx = 0; s_idx < static_cast<int>(all_S.size()); ++s_idx) {
        double best_for_S = -std::numeric_limits<double>::infinity();
        const std::vector<int>& S = all_S[s_idx];
        const std::vector<int> Sc = complement_indices(n, S);
        bool missing_for_S = false;

        for (int i : S) {
            double inner_best = std::numeric_limits<double>::infinity();
            bool found = false;

            for_each_subset_from_pool(Sc, k, [&](const std::vector<int>& I) {
                if (!invertible.count(subset_key(I))) return;
                const Eigen::MatrixXd GI = gather_columns(G, I);
                const Eigen::VectorXd coeffs = GI.colPivHouseholderQr().solve(G.col(i));
                inner_best = std::min(inner_best, coeffs.lpNorm<1>());
                found = true;
            });

            if (!found) {
                missing_invertible_subset = 1;
                missing_for_S = true;
                break;
            }
            best_for_S = std::max(best_for_S, inner_best);
        }

        if (!missing_for_S) best_outer = std::max(best_outer, best_for_S);
    }

    if (missing_invertible_subset) {
        throw std::runtime_error("No invertible I subset of S^c was found.");
    }
    return best_outer;
#endif
}

double h_m_roth_dual_combinatorial_parity(const Eigen::MatrixXd& H, int m, double tol) {
    validate_nonempty_matrix(H, "H");
    const int r = static_cast<int>(H.rows());
    const int n = static_cast<int>(H.cols());
    validate_m_for_columns(m, n);
    if (m > r) {
        throw std::invalid_argument("Parity-check form requires m <= r = n-k.");
    }

    std::vector<std::vector<int>> good_J;
    for_each_subset(n, r, [&](const std::vector<int>& J) {
        if (is_nonsingular(gather_columns(H, J), tol)) good_J.push_back(J);
    });

    double best_outer = -std::numeric_limits<double>::infinity();
    for_each_subset(n, m, [&](const std::vector<int>& S) {
        for (int i : S) {
            double inner_best = std::numeric_limits<double>::infinity();
            bool found = false;

            for (const auto& J : good_J) {
                if (!std::includes(J.begin(), J.end(), S.begin(), S.end())) continue;
                const int pos_i = position_in_sorted_subset(J, i);
                if (pos_i < 0) continue;

                const Eigen::MatrixXd HJ = gather_columns(H, J);
                const Eigen::RowVectorXd row_i = HJ.inverse().row(pos_i);
                const Eigen::MatrixXd HJc = gather_columns(H, complement_indices(n, J));
                const double val = (row_i * HJc).cwiseAbs().sum();

                inner_best = std::min(inner_best, val);
                found = true;
            }

            if (!found) {
                throw std::runtime_error("No invertible J superset of S was found.");
            }
            best_outer = std::max(best_outer, inner_best);
        }
    });
    return best_outer;
}

double h_m_roth_dual_combinatorial_parity_omp(const Eigen::MatrixXd& H, int m, double tol) {
    validate_nonempty_matrix(H, "H");
    const int r = static_cast<int>(H.rows());
    const int n = static_cast<int>(H.cols());
    validate_m_for_columns(m, n);
    if (m > r) {
        throw std::invalid_argument("Parity-check form requires m <= r = n-k.");
    }

#ifndef _OPENMP
    return h_m_roth_dual_combinatorial_parity(H, m, tol);
#else
    const std::vector<std::vector<int>> good_J = collect_nonsingular_subsets_parallel(H, r, tol);
    const std::vector<std::vector<int>> all_S = collect_subsets(n, m);

    double best_outer = -std::numeric_limits<double>::infinity();
    int missing_invertible_superset = 0;
#pragma omp parallel for schedule(dynamic) reduction(max: best_outer) reduction(|: missing_invertible_superset)
    for (int s_idx = 0; s_idx < static_cast<int>(all_S.size()); ++s_idx) {
        double best_for_S = -std::numeric_limits<double>::infinity();
        const std::vector<int>& S = all_S[s_idx];
        bool missing_for_S = false;

        for (int i : S) {
            double inner_best = std::numeric_limits<double>::infinity();
            bool found = false;

            for (const auto& J : good_J) {
                if (!std::includes(J.begin(), J.end(), S.begin(), S.end())) continue;
                const int pos_i = position_in_sorted_subset(J, i);
                if (pos_i < 0) continue;

                const Eigen::MatrixXd HJ = gather_columns(H, J);
                const Eigen::RowVectorXd row_i = HJ.inverse().row(pos_i);
                const Eigen::MatrixXd HJc = gather_columns(H, complement_indices(n, J));
                const double val = (row_i * HJc).cwiseAbs().sum();

                inner_best = std::min(inner_best, val);
                found = true;
            }

            if (!found) {
                missing_invertible_superset = 1;
                missing_for_S = true;
                break;
            }
            best_for_S = std::max(best_for_S, inner_best);
        }

        if (!missing_for_S) best_outer = std::max(best_outer, best_for_S);
    }

    if (missing_invertible_superset) {
        throw std::runtime_error("No invertible J superset of S was found.");
    }
    return best_outer;
#endif
}
