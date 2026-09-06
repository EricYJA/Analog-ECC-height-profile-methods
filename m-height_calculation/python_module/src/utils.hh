#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace m_height_utils {

inline void validate_num_threads(int num_threads) {
    if (num_threads < 1) {
        throw std::invalid_argument("num_threads must be at least 1.");
    }
}

inline void validate_m_for_columns(int m, int n) {
    if (m < 0 || m >= n) {
        throw std::invalid_argument("m must satisfy 0 <= m <= n-1.");
    }
}

namespace detail {

inline void validate_rank_tolerance(double tol) {
    if (!std::isfinite(tol) || tol < 0.0) {
        throw std::invalid_argument("tol must be a finite nonnegative value.");
    }
}

inline double matrix_scale(const Eigen::MatrixXd& M) {
    if (M.rows() == 0 || M.cols() == 0) return 0.0;
    return M.cwiseAbs().maxCoeff();
}

inline void set_numerical_rank_threshold(
    Eigen::FullPivLU<Eigen::MatrixXd>& lu,
    double tol,
    double reference_scale)
{
    const double max_pivot = lu.maxPivot();
    if (max_pivot == 0.0) {
        lu.setThreshold(1.0);
        return;
    }
    lu.setThreshold(tol * reference_scale / max_pivot);
}

inline int numerical_rank(
    const Eigen::MatrixXd& M,
    double tol,
    double reference_scale)
{
    if (M.rows() == 0 || M.cols() == 0) return 0;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    set_numerical_rank_threshold(lu, tol, reference_scale);
    return static_cast<int>(lu.rank());
}

inline int numerical_rank(const Eigen::MatrixXd& M, double tol) {
    return numerical_rank(M, tol, matrix_scale(M));
}

template <typename Fn>
bool for_each_subset_until(int n, int subset_size, Fn&& fn) {
    if (subset_size < 0 || subset_size > n) return true;

    std::vector<int> subset;
    subset.reserve(subset_size);

    std::function<bool(int, int)> dfs = [&](int start, int need) {
        if (need == 0) return fn(subset);

        for (int value = start; value <= n - need; ++value) {
            subset.push_back(value);
            const bool keep_going = dfs(value + 1, need - 1);
            subset.pop_back();
            if (!keep_going) return false;
        }
        return true;
    };

    return dfs(0, subset_size);
}

// All enumeration uses the same lexicographic iterator.
template <typename Fn>
void for_each_subset(int n, int size, Fn&& fn) {
    for_each_subset_until(n, size, [&](const std::vector<int>& subset) {
        fn(subset);
        return true;
    });
}

inline std::vector<std::vector<int>> collect_subsets(int n, int size) {
    std::vector<std::vector<int>> subsets;
    for_each_subset(n, size, [&](const auto& subset) { subsets.push_back(subset); });
    return subsets;
}

template <typename Fn>
void for_each_subset_from_pool(const std::vector<int>& pool, int size, Fn&& fn) {
    if (size < 0 || size > static_cast<int>(pool.size())) return;
    std::vector<int> subset(size);
    for_each_subset(static_cast<int>(pool.size()), size, [&](const auto& indices) {
        for (int i = 0; i < size; ++i) subset[i] = pool[indices[i]];
        fn(subset);
    });
}

inline Eigen::MatrixXd gather_columns(
    const Eigen::MatrixXd& M,
    const std::vector<int>& columns)
{
    Eigen::MatrixXd out(M.rows(), static_cast<int>(columns.size()));
    for (int idx = 0; idx < static_cast<int>(columns.size()); ++idx) {
        out.col(idx) = M.col(columns[idx]);
    }
    return out;
}

// Rank acceptance and solves use the same factorization and parent scale.
inline Eigen::FullPivLU<Eigen::MatrixXd> factor_columns(
    const Eigen::MatrixXd& M, const std::vector<int>& columns,
    double tol, double reference_scale)
{
    Eigen::FullPivLU<Eigen::MatrixXd> lu(gather_columns(M, columns));
    set_numerical_rank_threshold(lu, tol, reference_scale);
    return lu;
}

inline std::vector<int> complement_indices(
    int n,
    const std::vector<int>& subset)
{
    std::vector<char> in_subset(n, 0);
    for (int idx : subset) in_subset[idx] = 1;

    std::vector<int> complement;
    complement.reserve(n - static_cast<int>(subset.size()));
    for (int idx = 0; idx < n; ++idx) {
        if (!in_subset[idx]) complement.push_back(idx);
    }
    return complement;
}

inline bool generator_minimum_distance_exceeds_validated(
    const Eigen::MatrixXd& G,
    int m,
    double tol)
{
    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    if (n - m < k) return false;
    const double scale = matrix_scale(G);

    // A generator code has d > m exactly when puncturing any m coordinates
    // preserves dimension k: rank(G_{S^c}) = k for every |S| = m.
    return for_each_subset_until(n, m, [&](const std::vector<int>& S) {
        const std::vector<int> Sc = complement_indices(n, S);
        return numerical_rank(gather_columns(G, Sc), tol, scale) == k;
    });
}

inline bool parity_check_minimum_distance_exceeds_validated(
    const Eigen::MatrixXd& H, int m, double tol)
{
    if (m > H.rows()) return false;
    const double scale = matrix_scale(H);
    // d > m iff every m-column submatrix is independent.
    return for_each_subset_until(static_cast<int>(H.cols()), m, [&](const auto& S) {
        return numerical_rank(gather_columns(H, S), tol, scale) == m;
    });
}

inline int generator_minimum_distance_validated(const Eigen::MatrixXd& G, double tol) {
    const int redundancy = static_cast<int>(G.cols() - G.rows());
    // Use the same rank predicate as scalar calls, including near rank boundaries.
    for (int m = 1; m <= redundancy; ++m) {
        if (!generator_minimum_distance_exceeds_validated(G, m, tol)) return m;
    }
    return redundancy + 1;
}

}  // namespace detail

inline void validate_generator_matrix(const Eigen::MatrixXd& G, double tol) {
    detail::validate_rank_tolerance(tol);

    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    if (k == 0 || n == 0) {
        throw std::invalid_argument("G must be a non-empty generator matrix.");
    }
    if (!G.allFinite()) {
        throw std::invalid_argument("G must contain only finite values.");
    }
    if (k > n) {
        std::ostringstream oss;
        oss << "G must satisfy rows <= cols, got rows=" << k << ", cols=" << n;
        throw std::invalid_argument(oss.str());
    }
    if (detail::numerical_rank(G, tol) != k) {
        throw std::invalid_argument("G must have full row rank.");
    }
}

inline void validate_parity_check_matrix(const Eigen::MatrixXd& H, double tol) {
    detail::validate_rank_tolerance(tol);

    const int r = static_cast<int>(H.rows());
    const int n = static_cast<int>(H.cols());
    if (n == 0) {
        throw std::invalid_argument("H must have at least one column.");
    }
    if (!H.allFinite()) {
        throw std::invalid_argument("H must contain only finite values.");
    }
    if (r >= n) {
        std::ostringstream oss;
        oss << "H must satisfy rows < cols for a nontrivial code, got rows="
            << r << ", cols=" << n;
        throw std::invalid_argument(oss.str());
    }
    if (detail::numerical_rank(H, tol) != r) {
        throw std::invalid_argument("H must have full row rank.");
    }
}

inline bool generator_minimum_distance_exceeds(
    const Eigen::MatrixXd& G, int m, double tol)
{
    validate_generator_matrix(G, tol);
    validate_m_for_columns(m, static_cast<int>(G.cols()));
    return detail::generator_minimum_distance_exceeds_validated(G, m, tol);
}

inline bool parity_check_minimum_distance_exceeds(
    const Eigen::MatrixXd& H, int m, double tol)
{
    validate_parity_check_matrix(H, tol);
    validate_m_for_columns(m, static_cast<int>(H.cols()));
    return detail::parity_check_minimum_distance_exceeds_validated(H, m, tol);
}

inline int generator_minimum_distance(const Eigen::MatrixXd& G, double tol) {
    validate_generator_matrix(G, tol);
    return detail::generator_minimum_distance_validated(G, tol);
}

}  // namespace m_height_utils
