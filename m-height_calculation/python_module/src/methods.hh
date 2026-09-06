#pragma once

#include <Eigen/Dense>
#include <vector>
#include <optional>

// Non-owning view accepting both column-major C++ and row-major NumPy data.
using LpInput = Eigen::Ref<const Eigen::MatrixXd, 0,
                           Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

double h_m_jiang_simplified_lp_glpk_early_quit(const Eigen::MatrixXd& G, int m, double early_quit_threshold);

double h_m_jiang_original_lp_glpk(const Eigen::MatrixXd& G, int m);

#ifdef HAVE_HIGHS
double h_m_jiang_simplified_lp_highs_early_quit(const LpInput& G, int m, double early_quit_threshold, int num_threads = 16);
double h_m_jiang_original_lp_highs(const LpInput& G, int m, int num_threads = 16);
#endif

double h_m_roth_primal_lp_glpk(const Eigen::MatrixXd& G, int m, double early_quit_threshold);
double h_m_roth_dual_lp_glpk(const Eigen::MatrixXd& G, int m, double early_quit_threshold);
#ifdef HAVE_HIGHS
double h_m_roth_primal_lp_highs(const LpInput& G, int m, double early_quit_threshold, int num_threads = 16);
double h_m_roth_dual_lp_highs(const LpInput& G, int m, double early_quit_threshold, int num_threads = 16);
#endif
double h_m_roth_mds_combinatorial(const Eigen::MatrixXd& G, int m, double tol = 1e-10, int num_threads = 16);
double h_m_roth_primal_combinatorial(const Eigen::MatrixXd& G, int m, double tol = 1e-10, int num_threads = 16);
std::vector<double> h_m_roth_primal_combinatorial(const Eigen::MatrixXd& G, std::nullopt_t, double tol = 1e-10, int num_threads = 16);
double h_m_roth_primal_combinatorial_pruning(const Eigen::MatrixXd& G, int m, double tol = 1e-10, int num_threads = 16);
double h_m_roth_dual_combinatorial_generator(const Eigen::MatrixXd& G, int m, double tol = 1e-10, int num_threads = 16);
double h_m_roth_dual_combinatorial_parity(const Eigen::MatrixXd& H, int m, double tol = 1e-10, int num_threads = 16);
double h_m_roth_mds_combinatorial_parity(const Eigen::MatrixXd& H, int m, double tol = 1e-10, int num_threads = 16);
