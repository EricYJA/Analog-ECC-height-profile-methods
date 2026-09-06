#pragma once

#include <Eigen/Dense>
#include <glpk.h>
#ifdef HAVE_HIGHS
#include <highs/Highs.h>
#endif
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace m_height_lp {

inline std::mutex glpk_mutex;
constexpr double infinity = 1e20;
enum class Backend { Glpk, Highs };
enum class Status { Optimal, Infeasible, Unbounded };
struct Result { Status status; double value; };

// One workspace per caller/worker. GLPK creates/deletes each problem.
// HiGHS retains its instance and model buffers, but reloads each model without
// carrying a simplex basis between solves.
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
#ifdef HAVE_HIGHS
        if (backend == Backend::Highs) {
            highs_ = std::make_unique<Highs>();
            highs_->setOptionValue("output_flag", false);
            highs_->setOptionValue("threads", 1);
            highs_->setOptionValue("solver", "simplex");
            highs_->setOptionValue("simplex_strategy", 0);
            highs_->setOptionValue("presolve", "off");
            highs_->setOptionValue("scaling", "off");
            highs_->setOptionValue("infinite_bound", infinity);
            highs_->setOptionValue("infinite_cost", infinity);
            lp_.num_col_ = cols;
            lp_.num_row_ = rows;
            lp_.sense_ = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
            lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
            lp_.a_matrix_.start_.resize(rows + 1);
            lp_.a_matrix_.index_.resize(rows * cols);
            lp_.a_matrix_.value_.resize(rows * cols);
            for (int r = 0; r <= rows; ++r) lp_.a_matrix_.start_[r] = r * cols;
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) lp_.a_matrix_.index_[r * cols + c] = c;
            lp_.col_cost_.resize(cols);
            lp_.col_lower_.resize(cols);
            lp_.col_upper_.resize(cols);
            lp_.row_lower_.resize(rows);
            lp_.row_upper_.resize(rows);
        }
#else
        if (backend == Backend::Highs)
            throw std::runtime_error("HiGHS support was not built.");
#endif
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
#ifdef HAVE_HIGHS
        for (int c = 0; c < cols; ++c) {
            lp_.col_cost_[c] = cost[c];
            lp_.col_lower_[c] = col_lower[c];
            lp_.col_upper_[c] = col_upper[c];
        }
        for (int r = 0; r < rows; ++r) {
            lp_.row_lower_[r] = lower[r];
            lp_.row_upper_[r] = upper[r];
            for (int c = 0; c < cols; ++c) lp_.a_matrix_.value_[r * cols + c] = A(r, c);
        }
        if (highs_->clearModel() != HighsStatus::kOk ||
            highs_->clearSolver() != HighsStatus::kOk ||
            highs_->passModel(lp_) != HighsStatus::kOk ||
            highs_->run() != HighsStatus::kOk)
            throw std::runtime_error("HiGHS model solve failed.");
        const auto status = highs_->getModelStatus();
        if (status == HighsModelStatus::kOptimal)
            return {Status::Optimal, highs_->getInfo().objective_function_value};
        if (status == HighsModelStatus::kInfeasible) return {Status::Infeasible, 0.0};
        if (status == HighsModelStatus::kUnbounded) return {Status::Unbounded, 0.0};
        throw std::runtime_error("Unexpected HiGHS model status: " + highs_->modelStatusToString(status));
#else
        throw std::runtime_error("HiGHS support was not built.");
#endif
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
#ifdef HAVE_HIGHS
    std::unique_ptr<Highs> highs_;
    HighsLp lp_;
#endif
};
} // namespace m_height_lp
