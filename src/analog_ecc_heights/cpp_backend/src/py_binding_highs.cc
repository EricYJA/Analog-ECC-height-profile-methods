#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include "methods.hh"

namespace py = pybind11;

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

PYBIND11_MODULE(_highs, m) {
    py::module_::import("numpy");
    m.doc() = "Native highs height methods";
    m.def("h_m_jiang_simplified_lp_highs_early_quit", &h_m_jiang_simplified_lp_highs_early_quit_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"), py::arg("num_threads") = 16,
          "Simplified Jiang LP with HiGHS; returns min(h_m, threshold). Use infinity for a full sweep.");
    m.def("h_m_jiang_original_lp_highs", &h_m_jiang_original_lp_highs_np,
          py::arg("G"), py::arg("m"), py::arg("num_threads") = 16,
          "Solve m-height using Jiang original LP formulation with HiGHS backend");
    m.def("h_m_roth_primal_lp_highs", &h_m_roth_primal_lp_highs_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"), py::arg("num_threads") = 16,
          "Roth primal LP with highs; returns min(h_m, threshold). Use infinity for a full sweep.");
    m.def("h_m_roth_dual_lp_highs", &h_m_roth_dual_lp_highs_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"), py::arg("num_threads") = 16,
          "Roth dual LP with highs; returns min(h_m, threshold). Use infinity for a full sweep.");
}
