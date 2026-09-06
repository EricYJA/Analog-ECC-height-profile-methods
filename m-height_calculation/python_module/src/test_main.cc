#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <string>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "methods.hh"

int main() {
#ifdef HAVE_HIGHS
    // The Python bridge uses this strided map. Ref must keep the same storage
    // address rather than silently materializing a column-major input copy.
    const double numpy_data[] = {1., 0., 1., 0., 1., 1.};
    using InputStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<const Eigen::MatrixXd, 0, InputStride> numpy_view(
        numpy_data, 2, 3, InputStride(1, 3));
    LpInput input_ref(numpy_view);
    if (input_ref.data() != numpy_data || input_ref(1, 1) != 1. ||
        input_ref(0, 1) != 0.) {
        throw std::runtime_error("NumPy LP input view must not copy or transpose data.");
    }
#endif
    auto make_matrix = [](int rows, int cols, const std::vector<double>& values) {
        if (static_cast<int>(values.size()) != rows * cols) {
            throw std::runtime_error("Invalid matrix data length.");
        }
        Eigen::MatrixXd M(rows, cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                M(r, c) = values[static_cast<size_t>(r) * cols + c];
            }
        }
        return M;
    };

    Eigen::MatrixXd G(2, 3);
    G << 1.0, 0.0, 1.0,
         0.0, 1.0, 1.0;

    Eigen::MatrixXd H(1, 3);
    H << 1.0, 1.0, -1.0;

    const int m = 1;
    const double ref = 2.0;
    const double tol = 1e-9;

    auto check_close = [&](const char* name, double value) {
        std::cout << name << " = " << value << "\n";
        if (!std::isfinite(value) || std::abs(value - ref) > tol) {
            throw std::runtime_error(std::string(name) + " failed reference check.");
        }
    };

    auto check_expected = [&](const char* name, double value, double expected) {
        std::cout << name << " = " << value << "\n";
        if (!std::isfinite(value) || std::abs(value - expected) > tol) {
            throw std::runtime_error(std::string(name) + " failed reference check.");
        }
    };

    auto check_positive_infinity = [&](const char* name, double value) {
        std::cout << name << " = " << value << "\n";
        if (!std::isinf(value) || value < 0.0) {
            throw std::runtime_error(std::string(name) + " must return positive infinity.");
        }
    };

    auto expect_invalid_argument = [&](const char* name, auto&& fn) {
        try {
            fn();
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error(std::string(name) + " must reject the invalid input.");
    };

    const double no_cap = std::numeric_limits<double>::infinity();
    // Reuse the existing mathematical cases for every LP implementation.
    // Original Jiang has no threshold API; the other three formulations do.
    auto check_lp_case = [&](const char* case_name, const Eigen::MatrixXd& matrix,
                             int index, double expected) {
        struct LpMethod {
            std::string name;
            std::function<double(double)> solve;
            bool supports_cap;
        };
        std::vector<LpMethod> methods = {
            {"h_m_jiang_original_lp_glpk",
             [&](double) { return h_m_jiang_original_lp_glpk(matrix, index); }, false},
            {"h_m_jiang_simplified_lp_glpk_early_quit",
             [&](double cap) { return h_m_jiang_simplified_lp_glpk_early_quit(matrix, index, cap); }, true},
            {"h_m_roth_primal_lp_glpk",
             [&](double cap) { return h_m_roth_primal_lp_glpk(matrix, index, cap); }, true},
            {"h_m_roth_dual_lp_glpk",
             [&](double cap) { return h_m_roth_dual_lp_glpk(matrix, index, cap); }, true},
        };
#ifdef HAVE_HIGHS
        for (int threads : {1, 16}) {
            const std::string suffix = " (num_threads=" + std::to_string(threads) + ")";
            methods.push_back({"h_m_jiang_original_lp_highs" + suffix,
                [&, threads](double) { return h_m_jiang_original_lp_highs(matrix, index, threads); }, false});
            methods.push_back({"h_m_jiang_simplified_lp_highs_early_quit" + suffix,
                [&, threads](double cap) { return h_m_jiang_simplified_lp_highs_early_quit(matrix, index, cap, threads); }, true});
            methods.push_back({"h_m_roth_primal_lp_highs" + suffix,
                [&, threads](double cap) { return h_m_roth_primal_lp_highs(matrix, index, cap, threads); }, true});
            methods.push_back({"h_m_roth_dual_lp_highs" + suffix,
                [&, threads](double cap) { return h_m_roth_dual_lp_highs(matrix, index, cap, threads); }, true});
        }
#endif
        std::cout << "\nLP case: " << case_name << " (m=" << index << ")" << std::endl;
        for (const auto& method : methods) {
            std::vector<double> caps = {no_cap};
            if (method.supports_cap) {
                if (std::isfinite(expected)) {
                    // Below, at, and above the height: exercise both early
                    // stopping and completing a sweep without reaching a cap.
                    caps.insert(caps.end(), {expected / 2.0, expected, expected * 2.0});
                } else {
                    caps.push_back(1.5); // unbounded height must still be capped
                }
            }
            for (double cap : caps) {
                const double target = std::min(expected, cap);
                const double value = method.solve(cap);
                std::cout << "  " << method.name << " cap=" << cap
                          << " -> " << value << std::endl;
                // A relative tolerance accommodates the larger 10-column
                // references; explicitly reject NaN and incorrect infinities.
                const bool matches = std::isinf(target)
                    ? (std::isinf(value) && value > 0.0)
                    : (std::isfinite(value) &&
                       std::abs(value - target) <= 1e-8 * std::max(1.0, std::abs(target)));
                if (!matches) {
                    throw std::runtime_error(method.name + " failed " + case_name +
                        " (m=" + std::to_string(index) + ", cap=" + std::to_string(cap) +
                        ", expected=" + std::to_string(target) + ").");
                }
            }
        }
    };

    check_lp_case("initial [3,2]", G, m, ref);
    check_close("h_m_roth_primal_combinatorial", h_m_roth_primal_combinatorial(G, m));
    check_close("h_m_roth_mds_combinatorial", h_m_roth_mds_combinatorial(G, m));
    check_close("h_m_roth_mds_combinatorial_omp", h_m_roth_mds_combinatorial_omp(G, m));
    check_close("h_m_roth_primal_combinatorial_omp", h_m_roth_primal_combinatorial_omp(G, m));
    check_close("h_m_roth_primal_combinatorial_pruning_omp", h_m_roth_primal_combinatorial_pruning_omp(G, m));
    check_close("h_m_roth_dual_combinatorial_generator", h_m_roth_dual_combinatorial_generator(G, m));
    check_close("h_m_roth_dual_combinatorial_generator_omp", h_m_roth_dual_combinatorial_generator_omp(G, m));
    check_close("h_m_roth_dual_combinatorial_parity", h_m_roth_dual_combinatorial_parity(H, m));
    check_close("h_m_roth_dual_combinatorial_parity_omp", h_m_roth_dual_combinatorial_parity_omp(H, m));

    // Theorems 5 and 8 apply for m < d. This [4,2,2] code checks
    // both sides of that boundary: h_1 = 1 and h_2 = +infinity.
    Eigen::MatrixXd G_distance_2(2, 4);
    G_distance_2 << 1.0, 0.0, 1.0, 0.0,
                    0.0, 1.0, 0.0, 1.0;

    Eigen::MatrixXd H_distance_2(2, 4);
    H_distance_2 << 1.0, 0.0, -1.0, 0.0,
                    0.0, 1.0, 0.0, -1.0;

    check_expected(
        "h_m_roth_dual_combinatorial_generator(m=0)",
        h_m_roth_dual_combinatorial_generator(G_distance_2, 0),
        1.0);
    check_expected(
        "h_m_roth_dual_combinatorial_generator_omp(m=0)",
        h_m_roth_dual_combinatorial_generator_omp(G_distance_2, 0),
        1.0);
    check_expected(
        "h_m_roth_dual_combinatorial_parity(m=0)",
        h_m_roth_dual_combinatorial_parity(H_distance_2, 0),
        1.0);
    check_expected(
        "h_m_roth_dual_combinatorial_parity_omp(m=0)",
        h_m_roth_dual_combinatorial_parity_omp(H_distance_2, 0),
        1.0);

    check_lp_case("distance-2 boundary", G_distance_2, 1, 1.0);
    check_lp_case("distance-2 boundary", G_distance_2, 2, no_cap);
    check_lp_case("distance-2 boundary", G_distance_2, 3, no_cap);

    const std::vector<std::pair<const char*, double>> finite_boundary_values = {
        {"h_m_roth_primal_combinatorial(d=2,m=1)",
         h_m_roth_primal_combinatorial(G_distance_2, 1)},
        {"h_m_roth_primal_combinatorial_omp(d=2,m=1)",
         h_m_roth_primal_combinatorial_omp(G_distance_2, 1)},
        {"h_m_roth_primal_combinatorial_pruning_omp(d=2,m=1)",
         h_m_roth_primal_combinatorial_pruning_omp(G_distance_2, 1)},
        {"h_m_roth_dual_combinatorial_generator(d=2,m=1)",
         h_m_roth_dual_combinatorial_generator(G_distance_2, 1)},
        {"h_m_roth_dual_combinatorial_generator_omp(d=2,m=1)",
         h_m_roth_dual_combinatorial_generator_omp(G_distance_2, 1)},
        {"h_m_roth_dual_combinatorial_parity(d=2,m=1)",
         h_m_roth_dual_combinatorial_parity(H_distance_2, 1)},
        {"h_m_roth_dual_combinatorial_parity_omp(d=2,m=1)",
         h_m_roth_dual_combinatorial_parity_omp(H_distance_2, 1)},
    };
    for (const auto& [name, value] : finite_boundary_values) {
        check_expected(name, value, 1.0);
    }

    const std::vector<std::pair<const char*, double>> infinite_boundary_values = {
        {"h_m_roth_primal_combinatorial(d=2,m=2)",
         h_m_roth_primal_combinatorial(G_distance_2, 2)},
        {"h_m_roth_primal_combinatorial_omp(d=2,m=2)",
         h_m_roth_primal_combinatorial_omp(G_distance_2, 2)},
        {"h_m_roth_primal_combinatorial_pruning_omp(d=2,m=2)",
         h_m_roth_primal_combinatorial_pruning_omp(G_distance_2, 2)},
        {"h_m_roth_dual_combinatorial_generator(d=2,m=2)",
         h_m_roth_dual_combinatorial_generator(G_distance_2, 2)},
        {"h_m_roth_dual_combinatorial_generator_omp(d=2,m=2)",
         h_m_roth_dual_combinatorial_generator_omp(G_distance_2, 2)},
        {"h_m_roth_dual_combinatorial_parity(d=2,m=2)",
         h_m_roth_dual_combinatorial_parity(H_distance_2, 2)},
        {"h_m_roth_dual_combinatorial_parity_omp(d=2,m=2)",
         h_m_roth_dual_combinatorial_parity_omp(H_distance_2, 2)},
        {"h_m_roth_primal_combinatorial(d=2,m=3)",
         h_m_roth_primal_combinatorial(G_distance_2, 3)},
        {"h_m_roth_primal_combinatorial_omp(d=2,m=3)",
         h_m_roth_primal_combinatorial_omp(G_distance_2, 3)},
        {"h_m_roth_primal_combinatorial_pruning_omp(d=2,m=3)",
         h_m_roth_primal_combinatorial_pruning_omp(G_distance_2, 3)},
        {"h_m_roth_dual_combinatorial_generator(d=2,m=3)",
         h_m_roth_dual_combinatorial_generator(G_distance_2, 3)},
        {"h_m_roth_dual_combinatorial_generator_omp(d=2,m=3)",
         h_m_roth_dual_combinatorial_generator_omp(G_distance_2, 3)},
        {"h_m_roth_dual_combinatorial_parity(d=2,m=3)",
         h_m_roth_dual_combinatorial_parity(H_distance_2, 3)},
        {"h_m_roth_dual_combinatorial_parity_omp(d=2,m=3)",
         h_m_roth_dual_combinatorial_parity_omp(H_distance_2, 3)},
    };
    for (const auto& [name, value] : infinite_boundary_values) {
        check_positive_infinity(name, value);
    }

    const std::vector<double> distance_2_profile =
        h_m_roth_primal_combinatorial_omp_all(G_distance_2);
    if (distance_2_profile.size() != 2) {
        throw std::runtime_error("The [4,2,2] height profile must contain two entries.");
    }
    check_expected("h_m_roth_primal_combinatorial_omp_all(d=2,m=1)",
                   distance_2_profile[0], 1.0);
    check_positive_infinity("h_m_roth_primal_combinatorial_omp_all(d=2,m=2)",
                            distance_2_profile[1]);

    // The all-m cutoff must use the same numerical-rank predicate as scalar
    // calls; a tolerance-based hyperplane shortcut can otherwise infer d = 0.
    Eigen::MatrixXd G_numerical_distance(2, 3);
    G_numerical_distance << 0.0,  0.0,   1e-4,
                            1e-6, -3e-5, 1.0;
    check_positive_infinity(
        "h_m_roth_primal_combinatorial(numerical d=1,m=1)",
        h_m_roth_primal_combinatorial(G_numerical_distance, 1));
    const std::vector<double> numerical_distance_profile =
        h_m_roth_primal_combinatorial_omp_all(G_numerical_distance);
    if (numerical_distance_profile.size() != 1) {
        throw std::runtime_error("The numerical [3,2] profile must contain one entry.");
    }
    check_positive_infinity(
        "h_m_roth_primal_combinatorial_omp_all(numerical d=1,m=1)",
        numerical_distance_profile[0]);

    Eigen::MatrixXd H_roundoff_column(1, 3);
    H_roundoff_column << 1.0, 1e-14, 1.0;
    check_positive_infinity(
        "h_m_roth_dual_combinatorial_parity(roundoff-sized column)",
        h_m_roth_dual_combinatorial_parity(H_roundoff_column, 1));
    check_positive_infinity(
        "h_m_roth_dual_combinatorial_parity_omp(roundoff-sized column)",
        h_m_roth_dual_combinatorial_parity_omp(H_roundoff_column, 1));

    // A zero caller tolerance means exact floating-point rank. The solve must
    // not silently substitute Eigen's larger default QR threshold.
    Eigen::MatrixXd G_exact_rank(2, 3);
    G_exact_rank << 1.0, 1.0,   0.0,
                    0.0, 1e-16, 1.0;
    const double exact_rank_ref =
        h_m_roth_primal_combinatorial(G_exact_rank, 1, 0.0);
    const std::vector<std::pair<const char*, double>> exact_rank_values = {
        {"h_m_roth_dual_combinatorial_generator(tol=0)",
         h_m_roth_dual_combinatorial_generator(G_exact_rank, 1, 0.0)},
        {"h_m_roth_dual_combinatorial_generator_omp(tol=0)",
         h_m_roth_dual_combinatorial_generator_omp(G_exact_rank, 1, 0.0)},
    };
    for (const auto& [name, value] : exact_rank_values) {
        std::cout << name << " = " << value << "\n";
        const double scale = std::max(1.0, std::abs(exact_rank_ref));
        if (!std::isfinite(value) ||
            std::abs(value - exact_rank_ref) > tol * scale) {
            throw std::runtime_error(
                std::string(name) + " must honor the caller's rank tolerance.");
        }
    }


    // Theorem 5, Eq. (15), uses strict comparisons with 1. A hidden band
    // around 1 would incorrectly admit the I={0} candidate in this example
    // and return 1 + epsilon instead of the true h_1 = 1.
    constexpr double strict_unit_epsilon = 5e-13;
    constexpr double strict_unit_check_tol = 1e-14;
    Eigen::MatrixXd G_strict_unit_boundary(1, 3);
    G_strict_unit_boundary <<
        1.0, 1.0 + strict_unit_epsilon, 1.0 + strict_unit_epsilon;
    const std::vector<std::pair<const char*, double>> strict_unit_values = {
        {"h_m_roth_primal_combinatorial(strict unit boundary)",
         h_m_roth_primal_combinatorial(G_strict_unit_boundary, 1, 0.0)},
        {"h_m_roth_primal_combinatorial_omp(strict unit boundary)",
         h_m_roth_primal_combinatorial_omp(G_strict_unit_boundary, 1, 0.0)},
        {"h_m_roth_primal_combinatorial_pruning_omp(strict unit boundary)",
         h_m_roth_primal_combinatorial_pruning_omp(
             G_strict_unit_boundary, 1, 0.0)},
    };
    for (const auto& [name, value] : strict_unit_values) {
        std::cout << name << " = " << value << "\n";
        if (!std::isfinite(value) ||
            std::abs(value - 1.0) > strict_unit_check_tol) {
            throw std::runtime_error(
                std::string(name) + " must use Theorem 5's strict unit comparisons.");
        }
    }
    const std::vector<double> strict_unit_profile =
        h_m_roth_primal_combinatorial_omp_all(G_strict_unit_boundary, 0.0);
    if (strict_unit_profile.size() != 2 ||
        !std::isfinite(strict_unit_profile[0]) ||
        std::abs(strict_unit_profile[0] - 1.0) > strict_unit_check_tol) {
        throw std::runtime_error(
            "h_m_roth_primal_combinatorial_omp_all must use Theorem 5's "
            "strict unit comparisons.");
    }
    expect_invalid_argument(
        "h_m_roth_mds_combinatorial(non-MDS)",
        [&] { (void)h_m_roth_mds_combinatorial(G_distance_2, 2); });
    expect_invalid_argument(
        "h_m_roth_mds_combinatorial_omp(non-MDS)",
        [&] { (void)h_m_roth_mds_combinatorial_omp(G_distance_2, 2); });

    Eigen::MatrixXd G_singular_square(2, 2);
    G_singular_square << 1.0, 0.0,
                         0.0, 0.0;
    expect_invalid_argument(
        "h_m_roth_mds_combinatorial(rank-deficient square G)",
        [&] { (void)h_m_roth_mds_combinatorial(G_singular_square, 0); });
    expect_invalid_argument(
        "h_m_roth_mds_combinatorial_omp(rank-deficient square G)",
        [&] { (void)h_m_roth_mds_combinatorial_omp(G_singular_square, 0); });

    Eigen::MatrixXd G_rank_deficient(2, 3);
    G_rank_deficient << 1.0, 0.0, 1.0,
                        0.0, 0.0, 0.0;
    expect_invalid_argument(
        "h_m_roth_primal_combinatorial(rank-deficient G)",
        [&] { (void)h_m_roth_primal_combinatorial(G_rank_deficient, 1); });
    expect_invalid_argument(
        "h_m_roth_primal_combinatorial_omp(rank-deficient G)",
        [&] { (void)h_m_roth_primal_combinatorial_omp(G_rank_deficient, 1); });
    expect_invalid_argument(
        "h_m_roth_primal_combinatorial_omp_all(rank-deficient G)",
        [&] { (void)h_m_roth_primal_combinatorial_omp_all(G_rank_deficient); });
    expect_invalid_argument(
        "h_m_roth_primal_combinatorial_pruning_omp(rank-deficient G)",
        [&] { (void)h_m_roth_primal_combinatorial_pruning_omp(
            G_rank_deficient, 1); });
    expect_invalid_argument(
        "h_m_roth_dual_combinatorial_generator(rank-deficient G)",
        [&] { (void)h_m_roth_dual_combinatorial_generator(G_rank_deficient, 1); });
    expect_invalid_argument(
        "h_m_roth_dual_combinatorial_generator_omp(rank-deficient G)",
        [&] { (void)h_m_roth_dual_combinatorial_generator_omp(
            G_rank_deficient, 1); });

    Eigen::MatrixXd H_rank_deficient(2, 4);
    H_rank_deficient << 1.0, 0.0, 1.0, 0.0,
                        1.0, 0.0, 1.0, 0.0;
    expect_invalid_argument(
        "h_m_roth_dual_combinatorial_parity(rank-deficient H)",
        [&] { (void)h_m_roth_dual_combinatorial_parity(H_rank_deficient, 1); });
    expect_invalid_argument(
        "h_m_roth_dual_combinatorial_parity_omp(rank-deficient H)",
        [&] { (void)h_m_roth_dual_combinatorial_parity_omp(H_rank_deficient, 1); });

    Eigen::MatrixXd G_mds_4_2(2, 4);
    G_mds_4_2 << 1.0, 0.0, 1.0, 1.0,
                 0.0, 1.0, 1.0, 2.0;

    const int m_mds_4_2 = static_cast<int>(G_mds_4_2.cols() - G_mds_4_2.rows());
    const double ref_mds_4_2 = h_m_roth_primal_combinatorial(G_mds_4_2, m_mds_4_2);
    const double mds_value_4_2 = h_m_roth_mds_combinatorial(G_mds_4_2, m_mds_4_2);
    const double mds_value_4_2_omp = h_m_roth_mds_combinatorial_omp(G_mds_4_2, m_mds_4_2);
    std::cout << "h_m_roth_mds_combinatorial([4,2] MDS) = " << mds_value_4_2 << "\n";
    if (std::abs(mds_value_4_2 - ref_mds_4_2) > tol) {
        throw std::runtime_error("h_m_roth_mds_combinatorial disagrees with the exact primal combinatorial solver on an MDS case.");
    }
    std::cout << "h_m_roth_mds_combinatorial_omp([4,2] MDS) = " << mds_value_4_2_omp << "\n";
    if (std::abs(mds_value_4_2_omp - ref_mds_4_2) > tol) {
        throw std::runtime_error("h_m_roth_mds_combinatorial_omp disagrees with the exact primal combinatorial solver on an MDS case.");
    }

    // Keep the LU solve threshold consistent with the parent-matrix rank
    // convention even when elimination causes pivot growth.
    Eigen::MatrixXd G_near_singular_mds(3, 4);
    G_near_singular_mds <<
         0.435485271,    -0.797032707,     0.677813719,    -0.822772142,
         0.0345131995,   -1.0,             0.107746224,     0.745771858,
        -2.01912337e-10, -1.09673288e-10,  1.28596255e-10,  5.25175799e-11;
    const double near_singular_tol = 1e-10;
    const double near_singular_ref = h_m_roth_primal_combinatorial(
        G_near_singular_mds, 1, near_singular_tol);
    check_expected(
        "h_m_roth_mds_combinatorial(near-singular MDS)",
        h_m_roth_mds_combinatorial(G_near_singular_mds, 1, near_singular_tol),
        near_singular_ref);
    check_expected(
        "h_m_roth_mds_combinatorial_omp(near-singular MDS)",
        h_m_roth_mds_combinatorial_omp(G_near_singular_mds, 1, near_singular_tol),
        near_singular_ref);

    check_lp_case("[4,2] MDS", G_mds_4_2, m_mds_4_2, ref_mds_4_2);

    const std::vector<double> hm_all = h_m_roth_primal_combinatorial_omp_all(G);
    if (hm_all.size() != 1 || std::abs(hm_all[0] - ref) > tol) {
        throw std::runtime_error("h_m_roth_primal_combinatorial_omp_all failed reference check.");
    }

    // This case distinguishes Roth Eq. (15), including its admissibility
    // conditions, from an unrestricted max over the complement. The latter
    // would incorrectly return 8 for every m.
    Eigen::MatrixXd G_primal_complement(2, 5);
    G_primal_complement << 1.0, 0.0, 1.0, 2.0, -1.0,
                           0.0, 1.0, 2.0, -1.0, 3.0;
    const std::vector<double> complement_refs = {2.0, 4.0, 8.0};
    for (int m_case = 1; m_case <= static_cast<int>(complement_refs.size()); ++m_case) {
        const double expected = complement_refs[static_cast<size_t>(m_case - 1)];
        check_lp_case("Roth complement", G_primal_complement, m_case, expected);
        const std::vector<std::pair<const char*, double>> values = {
            {"h_m_roth_primal_combinatorial", h_m_roth_primal_combinatorial(
                G_primal_complement, m_case)},
            {"h_m_roth_primal_combinatorial_omp", h_m_roth_primal_combinatorial_omp(
                G_primal_complement, m_case)},
            {"h_m_roth_primal_combinatorial_pruning_omp",
             h_m_roth_primal_combinatorial_pruning_omp(
                 G_primal_complement, m_case)},
        };
        for (const auto& [name, value] : values) {
            if (std::abs(value - expected) > tol) {
                throw std::runtime_error(
                    std::string(name) + " failed Roth complement reference check.");
            }
        }
    }
    const std::vector<double> complement_all =
        h_m_roth_primal_combinatorial_omp_all(G_primal_complement);
    if (complement_all.size() != complement_refs.size()) {
        throw std::runtime_error(
            "h_m_roth_primal_combinatorial_omp_all returned an unexpected "
            "Roth complement output length.");
    }
    for (int idx = 0; idx < static_cast<int>(complement_refs.size()); ++idx) {
        if (std::abs(complement_all[idx] - complement_refs[idx]) > tol) {
            throw std::runtime_error(
                "h_m_roth_primal_combinatorial_omp_all failed Roth complement reference check.");
        }
    }

    const Eigen::MatrixXd G_10_7_3 = make_matrix(7, 10, {
        1.219, -0.028, 1.206, 0.604, 0.055, -2.649, 1.212, -2.646, -0.373, 0.004, 
        0.040, 1.183, -2.666, -0.387, -0.033, 0.070, -0.863, 0.044, 0.063, -2.537, 
        0.622, 0.039, -0.075, 0.123, 0.620, 0.663, -0.060, -0.825, 0.112, 1.151, 
        -0.815, -0.369, 0.065, -0.045, -0.345, -0.831, -2.596, -0.421, -2.608, -0.106, 
        -0.090, -0.813, -0.380, -0.838, -2.602, 1.133, 0.053, 0.008, 1.203, -0.794, 
        -2.583, 0.672, -0.864, -2.654, -0.834, -0.410, -0.385, 1.166, -0.113, 0.584, 
        -0.411, -2.657, 0.648, 1.183, 1.200, -0.084, 0.638, 0.543, -0.824, -0.364, 
    });

    const Eigen::MatrixXd G_10_5_5 = make_matrix(5, 10, {
        1.019, -0.005, 0.004, 0.0, 0.011, -0.284, -0.257, 1.023, 1.625, -1.34,
        -0.002, 1.007, -0.005, -0.012, 0.002, 0.542, -1.197, 0.834, 0.799, -0.305,
        0.002, 0.017, 1.016, -0.019, 0.009, -0.13, 0.339, 1.248, -0.916, -0.063,
        -0.001, -0.008, 0.015, 1.001, 0.008, 0.975, -1.226, -0.841, 0.916, 0.26,
        0.005, 0.007, 0.012, 0.004, 1.005, 2.346, -1.832, -0.194, -1.172, 0.64
    });

    const std::vector<double> hm_all_10_5_5 = h_m_roth_primal_combinatorial_omp_all(G_10_5_5);
    if (hm_all_10_5_5.size() != static_cast<size_t>(G_10_5_5.cols() - G_10_5_5.rows())) {
        throw std::runtime_error("h_m_roth_primal_combinatorial_omp_all returned an unexpected output length.");
    }
    for (int m_case = 1; m_case <= static_cast<int>(hm_all_10_5_5.size()); ++m_case) {
        const double scalar_value = h_m_roth_primal_combinatorial_omp(G_10_5_5, m_case);
        if (std::abs(hm_all_10_5_5[static_cast<size_t>(m_case - 1)] - scalar_value) > tol) {
            throw std::runtime_error("h_m_roth_primal_combinatorial_omp_all disagrees with the scalar OMP solver.");
        }
    }
    std::cout << "\nh_m_roth_primal_combinatorial_omp_all consistency check passed.\n";

    auto run_omp_case = [&](const char* case_name, const Eigen::MatrixXd& G_case, int m_case) {
        std::cout << "\n" << case_name << " (m = " << m_case << ")\n";
        const double hm_omp = h_m_roth_primal_combinatorial_omp(G_case, m_case);
        std::cout << "h_m_roth_primal_combinatorial_omp = " << hm_omp << "\n";
        check_lp_case(case_name, G_case, m_case, hm_omp);
    };

    run_omp_case("G_10_7_3", G_10_7_3, 3);
    run_omp_case("G_10_5_5", G_10_5_5, 5);

    double hm_mds_10_7_3 = h_m_roth_mds_combinatorial(G_10_7_3, 3);
    double hm_mds_10_5_5 = h_m_roth_mds_combinatorial(G_10_5_5, 5);
    
    std::cout << "\nh_m_roth_mds_combinatorial(G_10_7_3) = " << hm_mds_10_7_3 << "\n";
    std::cout << "\nh_m_roth_mds_combinatorial(G_10_5_5) = " << hm_mds_10_5_5 << "\n";

    const Eigen::MatrixXd G_untf_regression_16_3 = make_matrix(3, 16, {
        4.72841867e-01, -5.01380294e-01, -9.99249916e-01, 3.72892349e-01,
        -6.10245099e-01, 9.81076462e-01, -7.89570445e-01, 2.89092062e-01,
        -2.62949763e-01, 8.72601257e-02, 1.67205905e-01, -3.73039049e-01,
        -7.70565441e-01, 2.26130227e-01, 1.70008917e-01, 8.72487779e-01,
        -8.00248993e-01, -3.11563510e-01, 8.54645680e-04, 6.82106596e-01,
        5.36368031e-01, 7.64385791e-02, 5.10098677e-01, 3.36848136e-01,
        -8.99054646e-01, -4.93364547e-01, -4.69744473e-01, -1.08496707e-01,
        4.99087418e-01, 8.74927523e-01, 9.81793458e-01, 4.48049129e-01,
        3.68811764e-01, 8.07183982e-01, -3.87152986e-02, -6.29032502e-01,
        -5.83018228e-01, -1.77893560e-01, 3.41141982e-01, -8.96079859e-01,
        -3.50083084e-01, -8.65434627e-01, -8.66823116e-01, -9.21450125e-01,
        -3.96409700e-01, 4.28213674e-01, -8.47264696e-02, 1.94979621e-01,
    });
    const int m_untf_regression_16_3 = 13;
    const double untf_regression_tol = 1e-8;
    const double untf_regression_ref = h_m_roth_primal_combinatorial(
        G_untf_regression_16_3, m_untf_regression_16_3, untf_regression_tol);
    const double untf_regression_ref_omp = h_m_roth_primal_combinatorial_omp(
        G_untf_regression_16_3, m_untf_regression_16_3, untf_regression_tol);
    std::cout << "\nh_m_roth_primal_combinatorial(UNTF regression [16,3]) = "
              << untf_regression_ref << "\n";
    const double untf_regression_rel_tol = 1e-5;
    auto check_regression_close = [&](const char* name, double value) {
        const double scale = std::max(1.0, std::abs(untf_regression_ref));
        if (std::abs(value - untf_regression_ref) > untf_regression_rel_tol * scale) {
            throw std::runtime_error(std::string(name) + " disagrees on the UNTF [16,3] regression case.");
        }
    };
    check_regression_close("h_m_roth_primal_combinatorial_omp", untf_regression_ref_omp);


    const Eigen::MatrixXd G_untf_regression_16_5 = make_matrix(5, 16, {
        0.47068268, 0.28331455, -0.1406186, -0.74256218,
        0.86869295, 0.59127478, 0.433781, 0.5542207,
        0.20091914, -0.40715317, 0.17832094, 0.44173169,
        0.03034614, -0.47072134, 0.18208844, -0.19681347,
        0.5984937, 0.00880956, 0.45906589, 0.55907775,
        -0.34857628, -0.02241269, -0.58113544, 0.40454894,
        0.46978048, -0.46248111, 0.59823427, -0.57962036,
        0.08175572, -0.6525966, 0.19409093, -0.31043222,
        -0.21121164, 0.33516646, 0.18965746, 0.01932031,
        -0.01668952, 0.08875758, 0.31454849, -0.70695715,
        -0.82785581, -0.66390218, 0.48281011, -0.18487199,
        -0.83860079, -0.07172548, 0.34693609, -0.4215082,
        -0.56868153, -0.53560052, 0.71674354, -0.26054345,
        -0.34341736, -0.00909982, 0.33231729, -0.09710413,
        -0.08980761, -0.32225622, 0.14900116, 0.47034863,
        0.51123864, -0.54437894, -0.84463223, -0.37919162,
        0.22859149, -0.72142275, -0.46882157, 0.26034077,
        0.07518716, -0.80120603, 0.51452901, 0.14128675,
        0.21337091, 0.27517691, 0.59582754, 0.46206911,
        -0.16666999, 0.22593388, 0.30887581, -0.7371844,
    });
    const int m_untf_regression_16_5 = 11;
    const double untf_regression_16_5_ref = h_m_roth_primal_combinatorial(
        G_untf_regression_16_5, m_untf_regression_16_5, untf_regression_tol);
    const double untf_regression_16_5_ref_omp = h_m_roth_primal_combinatorial_omp(
        G_untf_regression_16_5, m_untf_regression_16_5, untf_regression_tol);
    std::cout << "\nh_m_roth_primal_combinatorial(UNTF regression [16,5]) = "
              << untf_regression_16_5_ref << "\n";
    auto check_regression_16_5_close = [&](const char* name, double value) {
        const double scale = std::max(1.0, std::abs(untf_regression_16_5_ref));
        if (std::abs(value - untf_regression_16_5_ref) > untf_regression_rel_tol * scale) {
            throw std::runtime_error(std::string(name) + " disagrees on the UNTF [16,5] regression case.");
        }
    };
    check_regression_16_5_close("h_m_roth_primal_combinatorial_omp", untf_regression_16_5_ref_omp);

    std::cout << "All LP and Roth m-height checks passed.\n";
    return 0;
}
