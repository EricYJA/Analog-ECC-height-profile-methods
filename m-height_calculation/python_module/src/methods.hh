#pragma once

#include <Eigen/Dense>
#include <vector>

double h_m_jiang_simplified_lp_glpk_early_quit(const Eigen::MatrixXd& G, int m, double early_quit_threshold);

double h_m_jiang_original_lp_glpk(const Eigen::MatrixXd& G, int m);

#ifdef HAVE_HIGHS
double h_m_jiang_simplified_lp_highs_early_quit(const Eigen::MatrixXd& G, int m, double early_quit_threshold);
double h_m_jiang_original_lp_highs(const Eigen::MatrixXd& G, int m);
#endif

double h_m_roth_primal_lp(const Eigen::MatrixXd& G, int m);
double h_m_roth_primal_lp_constraint(const Eigen::MatrixXd& G, int m);
double h_m_roth_dual_lp(const Eigen::MatrixXd& G, int m);
double h_m_roth_primal_combinatorial(const Eigen::MatrixXd& G, int m, double tol = 1e-10);
double h_m_roth_mds_combinatorial(const Eigen::MatrixXd& G, int m, double tol = 1e-10);
double h_m_roth_mds_combinatorial_omp(const Eigen::MatrixXd& G, int m, double tol = 1e-10);
double h_m_roth_primal_combinatorial_omp(const Eigen::MatrixXd& G, int m, double tol = 1e-10);
std::vector<double> h_m_roth_primal_combinatorial_omp_all(const Eigen::MatrixXd& G, double tol = 1e-10);
double h_m_roth_primal_combinatorial_pruning_omp(const Eigen::MatrixXd& G, int m, double tol = 1e-10);
double h_m_roth_dual_combinatorial_generator(const Eigen::MatrixXd& G, int m, double tol = 1e-10);
double h_m_roth_dual_combinatorial_generator_omp(const Eigen::MatrixXd& G, int m, double tol = 1e-10);
double h_m_roth_dual_combinatorial_parity(const Eigen::MatrixXd& H, int m, double tol = 1e-10);
double h_m_roth_dual_combinatorial_parity_omp(const Eigen::MatrixXd& H, int m, double tol = 1e-10);
