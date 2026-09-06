#include "methods.hh"
#include "utils.hh"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
namespace comb = m_height_utils::detail;

// The MDS specialization has no inner minimization. G solves against the
// complement basis and uses column l1 norms; H uses the S basis and row l1 norms.
double mds_height(const Eigen::MatrixXd& M, int m, double tol,
                  int num_threads, bool parity) {
    m_height_utils::validate_num_threads(num_threads);
    if (parity) m_height_utils::validate_parity_check_matrix(M, tol);
    else m_height_utils::validate_generator_matrix(M, tol);
    const int n = static_cast<int>(M.cols());
    const int r = parity ? static_cast<int>(M.rows()) : n - static_cast<int>(M.rows());
    if (m != r) throw std::invalid_argument("MDS form requires m equal to the code redundancy.");
    if (r == 0) return 0.0;

    const bool is_mds = parity
        ? comb::parity_check_minimum_distance_exceeds_validated(M, r, tol)
        : comb::generator_minimum_distance_exceeds_validated(M, r, tol);
    if (!is_mds) throw std::invalid_argument("The matrix must define an MDS code.");

    const auto all_S = comb::collect_subsets(n, r);
    const double scale = comb::matrix_scale(M);
    double best = -std::numeric_limits<double>::infinity();
#pragma omp parallel for if(num_threads > 1) num_threads(num_threads) schedule(dynamic) reduction(max: best)
    for (int idx = 0; idx < static_cast<int>(all_S.size()); ++idx) {
        const auto& S = all_S[idx];
        const auto Sc = comb::complement_indices(n, S);
        const auto lu = comb::factor_columns(M, parity ? S : Sc, tol, scale);
        const Eigen::MatrixXd coefficients = lu.solve(comb::gather_columns(M, parity ? Sc : S));
        const double value = parity
            ? coefficients.cwiseAbs().rowwise().sum().maxCoeff()
            : coefficients.cwiseAbs().colwise().sum().maxCoeff();
        best = std::max(best, value);
    }
    return best;
}
}  // namespace

double h_m_roth_mds_combinatorial(
    const Eigen::MatrixXd& G, int m, double tol, int num_threads) {
    return mds_height(G, m, tol, num_threads, false);
}

double h_m_roth_mds_combinatorial_parity(
    const Eigen::MatrixXd& H, int m, double tol, int num_threads) {
    return mds_height(H, m, tol, num_threads, true);
}
