#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include "methods.hh"

namespace py = pybind11;

// Thin wrappers converting NumPy -> Eigen and delegating to the backends
static double
h_m_jiang_lp_glpk_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in, int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm; // one copy
    return h_m_jiang_lp_glpk(G_cm, m);
}

static double
h_m_jiang_lp_glpk_early_quit_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double early_quit_threshold)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_jiang_lp_glpk_early_quit(G_cm, m, early_quit_threshold);
}

#ifdef HAVE_HIGHS
static double
h_m_jiang_lp_highs_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in, int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm; // one copy
    return h_m_jiang_lp_highs(G_cm, m);
}

static double
h_m_jiang_lp_highs_more_constraint_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in, int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_jiang_lp_highs_more_constraint(G_cm, m);
}

static double
h_m_jiang_lp_highs_early_quit_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double early_quit_threshold)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_jiang_lp_highs_early_quit(G_cm, m, early_quit_threshold);
}

static double
h_m_jiang_original_highs_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in, int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_jiang_original_highs(G_cm, m);
}
#endif // HAVE_HIGHS

static double h_m_roth_primal_lp_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
                                    int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_primal_lp(G_cm, m);
}

static double h_m_roth_primal_lp_constraint_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
                                               int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_primal_lp_constraint(G_cm, m);
}

static double h_m_roth_dual_lp_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
                                  int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_dual_lp(G_cm, m);
}

static double h_m_roth_primal_combinatorial_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_primal_combinatorial(G_cm, m, tol);
}

static double h_m_roth_mds_combinatorial_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_mds_combinatorial(G_cm, m, tol);
}

static double h_m_roth_mds_combinatorial_omp_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_mds_combinatorial_omp(G_cm, m, tol);
}

static double h_m_roth_primal_combinatorial_omp_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_primal_combinatorial_omp(G_cm, m, tol);
}

static std::vector<double> h_m_roth_primal_combinatorial_omp_all_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_primal_combinatorial_omp_all(G_cm, tol);
}

static double h_m_roth_primal_combinatorial_pruning_omp_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_primal_combinatorial_pruning_omp(G_cm, m, tol);
}

static double h_m_roth_dual_combinatorial_generator_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_dual_combinatorial_generator(G_cm, m, tol);
}

static double h_m_roth_dual_combinatorial_generator_omp_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_dual_combinatorial_generator_omp(G_cm, m, tol);
}

static double h_m_roth_dual_combinatorial_parity_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> H_in,
    int m,
    double tol)
{
    py::buffer_info info = H_in.request();
    if (info.ndim != 2) throw std::runtime_error("H must be 2D");
    const int r = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> H_rm(static_cast<double*>(info.ptr), r, n);
    Eigen::MatrixXd H_cm = H_rm;
    return h_m_roth_dual_combinatorial_parity(H_cm, m, tol);
}

static double h_m_roth_dual_combinatorial_parity_omp_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> H_in,
    int m,
    double tol)
{
    py::buffer_info info = H_in.request();
    if (info.ndim != 2) throw std::runtime_error("H must be 2D");
    const int r = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> H_rm(static_cast<double*>(info.ptr), r, n);
    Eigen::MatrixXd H_cm = H_rm;
    return h_m_roth_dual_combinatorial_parity_omp(H_cm, m, tol);
}

PYBIND11_MODULE(solve_m_height_cpp, m) {
    py::module_::import("numpy");
    m.doc() = "m-height solvers (GLPK or HiGHS backend) with OpenMP";

    m.def("h_m_jiang_lp_glpk",  &h_m_jiang_lp_glpk_np,
          "Solve m-height using GLPK backend");
    m.def("h_m_jiang_lp_glpk_early_quit",  &h_m_jiang_lp_glpk_early_quit_np,
          "Solve m-height using GLPK backend with a threshold-based early quit");

#ifdef HAVE_HIGHS
    m.def("h_m_jiang_lp_highs", &h_m_jiang_lp_highs_np,
          "Solve m-height using HiGHS backend");
    m.def("h_m_jiang_lp_highs_more_constraint", &h_m_jiang_lp_highs_more_constraint_np,
          "Solve m-height using HiGHS backend with additional X lower-bound constraints");
    m.def("h_m_jiang_lp_highs_early_quit", &h_m_jiang_lp_highs_early_quit_np,
          "Solve m-height using HiGHS backend with a threshold-based early quit");
    m.def("h_m_jiang_original_highs", &h_m_jiang_original_highs_np,
          "Solve m-height using Jiang original LP formulation with HiGHS backend");
#endif // HAVE_HIGHS

    m.def("h_m_roth_primal_lp", &h_m_roth_primal_lp_np,
          py::arg("G"), py::arg("m"),
          "Exact h_m(C) via primal LP characterization.");
    m.def("h_m_roth_primal_lp_constraint", &h_m_roth_primal_lp_constraint_np,
          py::arg("G"), py::arg("m"),
          "Exact h_m(C) via primal LP characterization with dominance constraints inside S.");
    m.def("h_m_roth_dual_lp", &h_m_roth_dual_lp_np,
          py::arg("G"), py::arg("m"),
          "Exact h_m(C) via dual LP characterization.");
    m.def("h_m_roth_primal_combinatorial", &h_m_roth_primal_combinatorial_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_m(C) via primal combinatorial characterization.");
    m.def("h_m_roth_mds_combinatorial", &h_m_roth_mds_combinatorial_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_r(C) for an MDS code via the generator-matrix characterization with m = r = n-k.");
    m.def("h_m_roth_mds_combinatorial_omp", &h_m_roth_mds_combinatorial_omp_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_r(C) for an MDS code via the generator-matrix characterization using OpenMP when available.");
    m.def("h_m_roth_primal_combinatorial_omp", &h_m_roth_primal_combinatorial_omp_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_m(C) via primal combinatorial characterization using OpenMP when available.");
    m.def("h_m_roth_primal_combinatorial_omp_all", &h_m_roth_primal_combinatorial_omp_all_np,
          py::arg("G"), py::arg("tol") = 1e-10,
          "Return [h_1(C), h_2(C), ..., h_{n-k}(C)] via the primal combinatorial characterization using OpenMP when available.");
    m.def("h_m_roth_primal_combinatorial_pruning_omp", &h_m_roth_primal_combinatorial_pruning_omp_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_m(C) via primal combinatorial characterization with branch-and-bound pruning using OpenMP when available.");
    m.def("h_m_roth_dual_combinatorial_generator", &h_m_roth_dual_combinatorial_generator_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_m(C) via dual combinatorial characterization (generator form).");
    m.def("h_m_roth_dual_combinatorial_generator_omp", &h_m_roth_dual_combinatorial_generator_omp_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_m(C) via dual combinatorial characterization (generator form) using OpenMP when available.");
    m.def("h_m_roth_dual_combinatorial_parity", &h_m_roth_dual_combinatorial_parity_np,
          py::arg("H"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_m(C) via dual combinatorial characterization (parity-check form).");
    m.def("h_m_roth_dual_combinatorial_parity_omp", &h_m_roth_dual_combinatorial_parity_omp_np,
          py::arg("H"), py::arg("m"), py::arg("tol") = 1e-10,
          "Exact h_m(C) via dual combinatorial characterization (parity-check form) using OpenMP when available.");

}
