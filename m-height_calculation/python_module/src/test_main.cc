#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "methods.hh"

int main() {
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

    Eigen::MatrixXd G_untf_3_2(2, 3);
    const double sqrt3_over_2 = std::sqrt(3.0) / 2.0;
    G_untf_3_2 << 1.0, -0.5, -0.5,
                  0.0, sqrt3_over_2, -sqrt3_over_2;

    const int m = 1;
    const double ref = 2.0;
    const double tol = 1e-9;

    auto check_close = [&](const char* name, double value) {
        std::cout << name << " = " << value << "\n";
        if (std::abs(value - ref) > tol) {
            throw std::runtime_error(std::string(name) + " failed reference check.");
        }
    };

    check_close("h_m_jiang_lp_glpk", h_m_jiang_lp_glpk(G, m));
#ifdef HAVE_HIGHS
    check_close("h_m_jiang_lp_highs", h_m_jiang_lp_highs(G, m));
    check_close("h_m_jiang_original_highs", h_m_jiang_original_highs(G, m));
#endif

    check_close("h_m_roth_primal_lp", h_m_roth_primal_lp(G, m));
    check_close("h_m_roth_primal_lp_constraint", h_m_roth_primal_lp_constraint(G, m));
    check_close("h_m_roth_dual_lp", h_m_roth_dual_lp(G, m));
    check_close("h_m_roth_primal_combinatorial", h_m_roth_primal_combinatorial(G, m));
    check_close("h_m_roth_mds_combinatorial", h_m_roth_mds_combinatorial(G, m));
    check_close("h_m_roth_mds_combinatorial_omp", h_m_roth_mds_combinatorial_omp(G, m));
    check_close("h_m_roth_untf_combinatorial", h_m_roth_untf_combinatorial(G_untf_3_2, m));
    check_close("h_m_roth_untf_combinatorial_omp", h_m_roth_untf_combinatorial_omp(G_untf_3_2, m));
    check_close("h_m_roth_primal_combinatorial_omp", h_m_roth_primal_combinatorial_omp(G, m));
    check_close("h_m_roth_primal_combinatorial_pruning_omp", h_m_roth_primal_combinatorial_pruning_omp(G, m));
    check_close("h_m_roth_dual_combinatorial_generator", h_m_roth_dual_combinatorial_generator(G, m));
    check_close("h_m_roth_dual_combinatorial_generator_omp", h_m_roth_dual_combinatorial_generator_omp(G, m));
    check_close("h_m_roth_dual_combinatorial_parity", h_m_roth_dual_combinatorial_parity(H, m));
    check_close("h_m_roth_dual_combinatorial_parity_omp", h_m_roth_dual_combinatorial_parity_omp(H, m));

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
#ifdef HAVE_HIGHS
    const double jiang_highs_mds_value_4_2 = h_m_jiang_lp_highs(G_mds_4_2, m_mds_4_2);
    const double jiang_original_highs_mds_value_4_2 = h_m_jiang_original_highs(G_mds_4_2, m_mds_4_2);
    std::cout << "h_m_jiang_lp_highs([4,2] MDS) = " << jiang_highs_mds_value_4_2 << "\n";
    std::cout << "h_m_jiang_original_highs([4,2] MDS) = " << jiang_original_highs_mds_value_4_2 << "\n";
    if (std::abs(jiang_highs_mds_value_4_2 - ref_mds_4_2) > tol) {
        throw std::runtime_error("h_m_jiang_lp_highs disagrees with the exact primal combinatorial solver on an MDS case.");
    }
    if (std::abs(jiang_original_highs_mds_value_4_2 - ref_mds_4_2) > tol) {
        throw std::runtime_error("h_m_jiang_original_highs disagrees with the exact primal combinatorial solver on an MDS case.");
    }
#endif

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

    auto run_omp_case = [](const char* case_name, const Eigen::MatrixXd& G_case, int m_case) {
        std::cout << "\n" << case_name << " (m = " << m_case << ")\n";
        const double hm_omp = h_m_roth_primal_combinatorial_omp(G_case, m_case);
        std::cout << "h_m_roth_primal_combinatorial_omp = " << hm_omp << "\n";
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
    const double untf_regression_value = h_m_roth_untf_combinatorial(
        G_untf_regression_16_3, m_untf_regression_16_3, untf_regression_tol);
    const double untf_regression_value_omp = h_m_roth_untf_combinatorial_omp(
        G_untf_regression_16_3, m_untf_regression_16_3, untf_regression_tol);
    std::cout << "\nh_m_roth_primal_combinatorial(UNTF regression [16,3]) = "
              << untf_regression_ref << "\n";
    std::cout << "h_m_roth_untf_combinatorial(UNTF regression [16,3]) = "
              << untf_regression_value << "\n";
    const double untf_regression_rel_tol = 1e-5;
    auto check_regression_close = [&](const char* name, double value) {
        const double scale = std::max(1.0, std::abs(untf_regression_ref));
        if (std::abs(value - untf_regression_ref) > untf_regression_rel_tol * scale) {
            throw std::runtime_error(std::string(name) + " disagrees on the UNTF [16,3] regression case.");
        }
    };
    check_regression_close("h_m_roth_primal_combinatorial_omp", untf_regression_ref_omp);
    check_regression_close("h_m_roth_untf_combinatorial", untf_regression_value);
    check_regression_close("h_m_roth_untf_combinatorial_omp", untf_regression_value_omp);


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
    const double untf_regression_16_5_value = h_m_roth_untf_combinatorial(
        G_untf_regression_16_5, m_untf_regression_16_5, untf_regression_tol);
    const double untf_regression_16_5_value_omp = h_m_roth_untf_combinatorial_omp(
        G_untf_regression_16_5, m_untf_regression_16_5, untf_regression_tol);
    std::cout << "\nh_m_roth_primal_combinatorial(UNTF regression [16,5]) = "
              << untf_regression_16_5_ref << "\n";
    std::cout << "h_m_roth_untf_combinatorial(UNTF regression [16,5]) = "
              << untf_regression_16_5_value << "\n";
    auto check_regression_16_5_close = [&](const char* name, double value) {
        const double scale = std::max(1.0, std::abs(untf_regression_16_5_ref));
        if (std::abs(value - untf_regression_16_5_ref) > untf_regression_rel_tol * scale) {
            throw std::runtime_error(std::string(name) + " disagrees on the UNTF [16,5] regression case.");
        }
    };
    check_regression_16_5_close("h_m_roth_primal_combinatorial_omp", untf_regression_16_5_ref_omp);
    check_regression_16_5_close("h_m_roth_untf_combinatorial", untf_regression_16_5_value);
    check_regression_16_5_close("h_m_roth_untf_combinatorial_omp", untf_regression_16_5_value_omp);

    std::cout << "All Roth m-height checks passed.\n";
    return 0;
}
