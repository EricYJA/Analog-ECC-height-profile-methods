#pragma once

#include "lp_common.hh"
#include <Eigen/Dense>
#include <glpk.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace m_height_lp {

inline std::mutex glpk_mutex;
// GLPK workspace: each solve creates and deletes its own problem.
class Workspace {
public:
    Workspace(int rows, int cols, Backend backend, bool maximize)
        : A(rows, cols), cost(cols), lower(rows), upper(rows),
          col_lower(cols), col_upper(cols), backend_(backend), maximize_(maximize) {
        A.setZero();
        cost.setZero();
        lower.setConstant(-infinity);
        upper.setConstant(infinity);
        col_lower.setConstant(-infinity);
        col_upper.setConstant(infinity);
        if (backend != Backend::Glpk)
            throw std::invalid_argument("Use HighsWorkspace for HiGHS.");

    }

    Result solve() {
        const int rows = A.rows(), cols = A.cols();
        if (backend_ == Backend::Glpk) {
            std::lock_guard<std::mutex> lock(glpk_mutex);
            std::unique_ptr<glp_prob, decltype(&glp_delete_prob)> lp(
                glp_create_prob(), &glp_delete_prob);
            if (!lp) throw std::runtime_error("GLPK allocation failed.");
            glp_set_obj_dir(lp.get(), maximize_ ? GLP_MAX : GLP_MIN);
            glp_add_rows(lp.get(), rows);
            glp_add_cols(lp.get(), cols);
            for (int c = 0; c < cols; ++c) {
                glp_set_col_bnds(lp.get(), c + 1, bound_type(col_lower[c], col_upper[c]),
                                 col_lower[c], col_upper[c]);
                glp_set_obj_coef(lp.get(), c + 1, cost[c]);
            }
            std::vector<int> indices(cols + 1);
            std::vector<double> values(cols + 1);
            for (int c = 0; c < cols; ++c) indices[c + 1] = c + 1;
            for (int r = 0; r < rows; ++r) {
                glp_set_row_bnds(lp.get(), r + 1, bound_type(lower[r], upper[r]), lower[r], upper[r]);
                for (int c = 0; c < cols; ++c) values[c + 1] = A(r, c);
                glp_set_mat_row(lp.get(), r + 1, cols, indices.data(), values.data());
            }
            glp_smcp options;
            glp_init_smcp(&options);
            options.msg_lev = GLP_MSG_OFF;
            const int result = glp_simplex(lp.get(), &options);
            if (result != 0)
                throw std::runtime_error("GLPK simplex failed: " + std::to_string(result));
            switch (glp_get_status(lp.get())) {
                case GLP_OPT: return {Status::Optimal, glp_get_obj_val(lp.get())};
                case GLP_NOFEAS: return {Status::Infeasible, 0.0};
                case GLP_UNBND: return {Status::Unbounded, 0.0};
                default: throw std::runtime_error("Unexpected GLPK model status.");
            }
        }
        throw std::runtime_error("Invalid GLPK backend.");
    }

    Eigen::MatrixXd A;
    Eigen::VectorXd cost, lower, upper, col_lower, col_upper;

private:
    static int bound_type(double lower, double upper) {
        if (lower <= -infinity) return upper >= infinity ? GLP_FR : GLP_UP;
        if (upper >= infinity) return GLP_LO;
        return lower == upper ? GLP_FX : GLP_DB;
    }
    Backend backend_;
    bool maximize_;
};
} // namespace m_height_lp

