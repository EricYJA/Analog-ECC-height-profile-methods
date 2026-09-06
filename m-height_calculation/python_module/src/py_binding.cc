#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include "methods.hh"

namespace py = pybind11;

// Thin wrappers converting NumPy -> Eigen and delegating to the backends
static double
h_m_jiang_original_lp_glpk_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in, int m)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm; // one copy
    return h_m_jiang_original_lp_glpk(G_cm, m);
}

static double
h_m_jiang_simplified_lp_glpk_early_quit_np(
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
    return h_m_jiang_simplified_lp_glpk_early_quit(G_cm, m, early_quit_threshold);
}

#ifdef HAVE_HIGHS
static double
h_m_jiang_simplified_lp_highs_early_quit_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double early_quit_threshold, int num_threads)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    // Column-major view with explicit strides over row-major NumPy storage.
    // Matching storage-order traits prevents Eigen::Ref from making a copy.
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<const Eigen::MatrixXd, 0, Stride> G_rm(
        static_cast<double*>(info.ptr), info.shape[0], info.shape[1],
        Stride(1, info.shape[1]));
    return h_m_jiang_simplified_lp_highs_early_quit(G_rm, m, early_quit_threshold, num_threads);
}

static double
h_m_jiang_original_lp_highs_np(py::array_t<double, py::array::c_style | py::array::forcecast> G_in, int m, int num_threads)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    // Column-major view with explicit strides over row-major NumPy storage.
    // Matching storage-order traits prevents Eigen::Ref from making a copy.
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<const Eigen::MatrixXd, 0, Stride> G_rm(
        static_cast<double*>(info.ptr), info.shape[0], info.shape[1],
        Stride(1, info.shape[1]));
    return h_m_jiang_original_lp_highs(G_rm, m, num_threads);
}
#endif // HAVE_HIGHS

static double h_m_roth_primal_lp_glpk_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m, double early_quit_threshold)
{
    const auto info = G_in.request();
    if (info.ndim != 2) throw std::invalid_argument("G must be 2D");
    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), info.shape[0], info.shape[1]);
    const Eigen::MatrixXd G = G_rm;
    return h_m_roth_primal_lp_glpk(G, m, early_quit_threshold);
}

static double h_m_roth_dual_lp_glpk_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m, double early_quit_threshold)
{
    const auto info = G_in.request();
    if (info.ndim != 2) throw std::invalid_argument("G must be 2D");
    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), info.shape[0], info.shape[1]);
    const Eigen::MatrixXd G = G_rm;
    return h_m_roth_dual_lp_glpk(G, m, early_quit_threshold);
}

#ifdef HAVE_HIGHS
static double h_m_roth_primal_lp_highs_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m, double early_quit_threshold, int num_threads)
{
    const auto info = G_in.request();
    if (info.ndim != 2) throw std::invalid_argument("G must be 2D");
    // Column-major view with explicit strides over row-major NumPy storage.
    // Matching storage-order traits prevents Eigen::Ref from making a copy.
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<const Eigen::MatrixXd, 0, Stride> G_rm(
        static_cast<double*>(info.ptr), info.shape[0], info.shape[1],
        Stride(1, info.shape[1]));
    return h_m_roth_primal_lp_highs(G_rm, m, early_quit_threshold, num_threads);
}

static double h_m_roth_dual_lp_highs_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m, double early_quit_threshold, int num_threads)
{
    const auto info = G_in.request();
    if (info.ndim != 2) throw std::invalid_argument("G must be 2D");
    // Column-major view with explicit strides over row-major NumPy storage.
    // Matching storage-order traits prevents Eigen::Ref from making a copy.
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<const Eigen::MatrixXd, 0, Stride> G_rm(
        static_cast<double*>(info.ptr), info.shape[0], info.shape[1],
        Stride(1, info.shape[1]));
    return h_m_roth_dual_lp_highs(G_rm, m, early_quit_threshold, num_threads);
}

#endif

static double h_m_roth_mds_combinatorial_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol, int num_threads)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_mds_combinatorial(G_cm, m, tol, num_threads);
}

static py::object h_m_roth_primal_combinatorial_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    std::optional<int> m,
    double tol, int num_threads)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    if (m.has_value()) {
        return py::cast(h_m_roth_primal_combinatorial(G_cm, *m, tol, num_threads));
    }
    return py::cast(h_m_roth_primal_combinatorial(
        G_cm, std::nullopt, tol, num_threads));
}

static double h_m_roth_primal_combinatorial_pruning_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol, int num_threads)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_primal_combinatorial_pruning(G_cm, m, tol, num_threads);
}

static double h_m_roth_dual_combinatorial_generator_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    int m,
    double tol, int num_threads)
{
    py::buffer_info info = G_in.request();
    if (info.ndim != 2) throw std::runtime_error("G must be 2D");
    const int k = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> G_rm(static_cast<double*>(info.ptr), k, n);
    Eigen::MatrixXd G_cm = G_rm;
    return h_m_roth_dual_combinatorial_generator(G_cm, m, tol, num_threads);
}

static double h_m_roth_dual_combinatorial_parity_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> H_in,
    int m,
    double tol, int num_threads)
{
    py::buffer_info info = H_in.request();
    if (info.ndim != 2) throw std::runtime_error("H must be 2D");
    const int r = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> H_rm(static_cast<double*>(info.ptr), r, n);
    Eigen::MatrixXd H_cm = H_rm;
    return h_m_roth_dual_combinatorial_parity(H_cm, m, tol, num_threads);
}

static double h_m_roth_mds_combinatorial_parity_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> H_in,
    int m,
    double tol, int num_threads)
{
    py::buffer_info info = H_in.request();
    if (info.ndim != 2) throw std::runtime_error("H must be 2D");
    const int r = (int)info.shape[0], n = (int)info.shape[1];

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> H_rm(static_cast<double*>(info.ptr), r, n);
    Eigen::MatrixXd H_cm = H_rm;
    return h_m_roth_mds_combinatorial_parity(H_cm, m, tol, num_threads);
}

PYBIND11_MODULE(solve_m_height_cpp, m) {
    py::module_::import("numpy");
    m.doc() = "m-height solvers (GLPK or HiGHS backend) with OpenMP";

    m.def("h_m_jiang_original_lp_glpk", &h_m_jiang_original_lp_glpk_np,
          py::arg("G"), py::arg("m"),
          "Solve m-height using Jiang original LP formulation with GLPK backend.");
    m.def("h_m_jiang_simplified_lp_glpk_early_quit",  &h_m_jiang_simplified_lp_glpk_early_quit_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"),
          "Simplified Jiang LP with GLPK; returns min(h_m, threshold). Use infinity for a full sweep.");

#ifdef HAVE_HIGHS
    m.def("h_m_jiang_simplified_lp_highs_early_quit", &h_m_jiang_simplified_lp_highs_early_quit_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"), py::arg("num_threads") = 16,
          "Simplified Jiang LP with HiGHS; returns min(h_m, threshold). Use infinity for a full sweep.");
    m.def("h_m_jiang_original_lp_highs", &h_m_jiang_original_lp_highs_np,
          py::arg("G"), py::arg("m"), py::arg("num_threads") = 16,
          "Solve m-height using Jiang original LP formulation with HiGHS backend");
#endif // HAVE_HIGHS

    m.def("h_m_roth_primal_lp_glpk", &h_m_roth_primal_lp_glpk_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"),
          "Roth primal LP with glpk; returns min(h_m, threshold). Use infinity for a full sweep.");
    m.def("h_m_roth_dual_lp_glpk", &h_m_roth_dual_lp_glpk_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"),
          "Roth dual LP with glpk; returns min(h_m, threshold). Use infinity for a full sweep.");
#ifdef HAVE_HIGHS
    m.def("h_m_roth_primal_lp_highs", &h_m_roth_primal_lp_highs_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"), py::arg("num_threads") = 16,
          "Roth primal LP with highs; returns min(h_m, threshold). Use infinity for a full sweep.");
    m.def("h_m_roth_dual_lp_highs", &h_m_roth_dual_lp_highs_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"), py::arg("num_threads") = 16,
          "Roth dual LP with highs; returns min(h_m, threshold). Use infinity for a full sweep.");
#endif
    m.def("h_m_roth_mds_combinatorial", &h_m_roth_mds_combinatorial_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10, py::arg("num_threads") = 16,
          "Exact h_r(C) for an MDS code via the generator-matrix characterization using OpenMP when available.");
    m.def("h_m_roth_primal_combinatorial", &h_m_roth_primal_combinatorial_np,
          py::arg("G"), py::arg("m") = py::none(), py::arg("tol") = 1e-10, py::arg("num_threads") = 16,
          "Return scalar h_m for integer m, or [h_1, ..., h_{n-k}] when m is None or omitted.");
    m.def("h_m_roth_primal_combinatorial_pruning", &h_m_roth_primal_combinatorial_pruning_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10, py::arg("num_threads") = 16,
          "Exact h_m(C) via primal combinatorial characterization with branch-and-bound pruning using OpenMP when available.");
    m.def("h_m_roth_dual_combinatorial_generator", &h_m_roth_dual_combinatorial_generator_np,
          py::arg("G"), py::arg("m"), py::arg("tol") = 1e-10, py::arg("num_threads") = 16,
          "Exact h_m(C) via dual combinatorial characterization (generator form) using OpenMP when available.");
    m.def("h_m_roth_dual_combinatorial_parity", &h_m_roth_dual_combinatorial_parity_np,
          py::arg("H"), py::arg("m"), py::arg("tol") = 1e-10, py::arg("num_threads") = 16,
          "Exact h_m(C) via dual combinatorial characterization (parity-check form) using OpenMP when available.");

    m.def("h_m_roth_mds_combinatorial_parity", &h_m_roth_mds_combinatorial_parity_np,
          py::arg("H"), py::arg("m"), py::arg("tol") = 1e-10, py::arg("num_threads") = 16,
          "Exact h_r(C) for an MDS code via its parity-check matrix, with m=r=H.rows().");

}
