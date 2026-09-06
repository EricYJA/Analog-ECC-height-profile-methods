#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include "methods.hh"

namespace py = pybind11;

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

PYBIND11_MODULE(_glpk, m) {
    py::module_::import("numpy");
    m.doc() = "Native glpk height methods";
    m.def("h_m_jiang_original_lp_glpk", &h_m_jiang_original_lp_glpk_np,
          py::arg("G"), py::arg("m"),
          "Solve m-height using Jiang original LP formulation with GLPK backend.");
    m.def("h_m_jiang_simplified_lp_glpk_early_quit",  &h_m_jiang_simplified_lp_glpk_early_quit_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"),
          "Simplified Jiang LP with GLPK; returns min(h_m, threshold). Use infinity for a full sweep.");
    m.def("h_m_roth_primal_lp_glpk", &h_m_roth_primal_lp_glpk_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"),
          "Roth primal LP with glpk; returns min(h_m, threshold). Use infinity for a full sweep.");
    m.def("h_m_roth_dual_lp_glpk", &h_m_roth_dual_lp_glpk_np,
          py::arg("G"), py::arg("m"), py::arg("early_quit_threshold"),
          "Roth dual LP with glpk; returns min(h_m, threshold). Use infinity for a full sweep.");
}
