#pragma once

#include <Eigen/Dense>
#include <glpk.h>
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

#ifdef HAVE_HIGHS

#include "methods.hh"
#include <highs/Highs.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace m_height_lp {

// Direct row-wise storage: builders write into the arrays passed to HiGHS.
class HighsWorkspace {
public:
    HighsWorkspace(int rows, int cols, bool maximize) : cols_(cols) {
        option("output_flag", false);
        option("threads", 1);
        option("parallel", "off");
        option("solver", "simplex");
        option("simplex_strategy", 0);
        option("simplex_scale_strategy", 0);
        option("presolve", "off");
        option("infinite_bound", infinity);
        option("infinite_cost", infinity);
        lp.num_col_ = cols;
        lp.num_row_ = rows;
        lp.sense_ = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
        lp.col_cost_.assign(cols, 0.0);
        lp.col_lower_.assign(cols, -infinity);
        lp.col_upper_.assign(cols, infinity);
        lp.row_lower_.assign(rows, -infinity);
        lp.row_upper_.assign(rows, infinity);
        lp.a_matrix_.format_ = MatrixFormat::kRowwise;
        lp.a_matrix_.start_.resize(rows + 1);
        lp.a_matrix_.index_.resize(static_cast<size_t>(rows) * cols);
        lp.a_matrix_.value_.resize(static_cast<size_t>(rows) * cols);
        for (int r = 0; r <= rows; ++r) lp.a_matrix_.start_[r] = static_cast<HighsInt>(r) * cols;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                lp.a_matrix_.index_[static_cast<size_t>(r) * cols + c] = c;
    }

    double& coefficient(int row, int col) {
        return lp.a_matrix_.value_[static_cast<size_t>(row) * cols_ + col];
    }

    Result solve() {
        if (highs_.clearModel() != HighsStatus::kOk ||
            highs_.clearSolver() != HighsStatus::kOk ||
            highs_.passModel(lp) != HighsStatus::kOk ||
            highs_.run() != HighsStatus::kOk)
            throw std::runtime_error("HiGHS model solve failed.");
        const auto status = highs_.getModelStatus();
        if (status == HighsModelStatus::kOptimal)
            return {Status::Optimal, highs_.getInfo().objective_function_value};
        if (status == HighsModelStatus::kInfeasible) return {Status::Infeasible, 0.0};
        if (status == HighsModelStatus::kUnbounded) return {Status::Unbounded, 0.0};
        throw std::runtime_error("Unexpected HiGHS model status: " + highs_.modelStatusToString(status));
    }

    HighsLp lp;
private:
    template<typename T> void option(const char* name, T value) {
        if (highs_.setOptionValue(name, value) != HighsStatus::kOk)
            throw std::runtime_error(std::string("HiGHS rejected option: ") + name);
    }
    int cols_;
    Highs highs_;
};

// Lexicographic combinations, O(size) state rather than a full task list.
class Combinations {
public:
    Combinations(int n, int size) : n_(n), values(size) {
        std::iota(values.begin(), values.end(), 0);
    }
    bool advance() {
        for (int t = static_cast<int>(values.size()) - 1; t >= 0; --t) {
            if (values[t] < n_ - static_cast<int>(values.size()) + t) {
                ++values[t];
                for (int j = t + 1; j < static_cast<int>(values.size()); ++j)
                    values[j] = values[j - 1] + 1;
                return true;
            }
        }
        return false;
    }
    int n_;
    std::vector<int> values;
};

inline void validate_highs_input(const LpInput& G, int m, int num_threads, double threshold) {
    if (G.rows() == 0 || G.cols() == 0 || !G.allFinite())
        throw std::invalid_argument("G must be a finite, nonempty matrix.");
    if (m < 0 || m >= G.cols()) throw std::invalid_argument("m must satisfy 0 <= m <= n-1.");
    if (num_threads < 1) throw std::invalid_argument("num_threads must be at least 1.");
    if (std::isnan(threshold)) throw std::invalid_argument("early_quit_threshold must not be NaN.");
}

// next(job) lazily produces cases. evaluate() must use completed LP results.
// Persistent worker region keeps one workspace per worker for the whole call.
// Only 4*num_threads jobs are buffered, independent of total search size.
template<typename Job, typename Next, typename Evaluate>
double highs_sweep(int rows, int cols, bool maximize, int num_threads,
                   double threshold, Next&& next, Evaluate&& evaluate) {
    if (num_threads < 1) throw std::invalid_argument("num_threads must be at least 1.");
    int workers = num_threads;
#ifndef _OPENMP
    workers = 1;
#endif
    if (workers == 1) {
        HighsWorkspace workspace(rows, cols, maximize);
        double best = -std::numeric_limits<double>::infinity();
        Job job;
        while (next(job)) {
            best = std::max(best, evaluate(workspace, job, [] { return false; }));
            if (best > threshold || best == std::numeric_limits<double>::infinity()) break;
        }
        return std::min(best, threshold);
    }

    std::vector<Job> batch;
    std::atomic<bool> stop{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    double best = -std::numeric_limits<double>::infinity();
    auto fail = [&] {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error) error = std::current_exception();
        stop.store(true);
    };
#pragma omp parallel num_threads(workers) shared(batch, stop, error, best)
    {
        std::unique_ptr<HighsWorkspace> workspace;
        double local_best = -std::numeric_limits<double>::infinity();
        try { workspace = std::make_unique<HighsWorkspace>(rows, cols, maximize); }
        catch (...) { fail(); }
#pragma omp barrier
        while (true) {
#pragma omp single
            {
                batch.clear();
                if (!stop.load()) {
                    try {
                        Job job;
                        const size_t capacity = static_cast<size_t>(workers) * 4;
                        while (batch.size() < capacity && next(job)) batch.push_back(job);
                    } catch (...) { fail(); batch.clear(); }
                }
            }
            // All workers observe the same batch after the single barrier.
            if (batch.empty()) break;
#pragma omp for schedule(dynamic, 1)
            for (int t = 0; t < static_cast<int>(batch.size()); ++t) {
                if (stop.load()) continue;
                try {
                    const double value = evaluate(*workspace, batch[t], [&] { return stop.load(); });
                    local_best = std::max(local_best, value);
                    if (value > threshold || value == std::numeric_limits<double>::infinity())
                        stop.store(true);
                } catch (...) { fail(); }
            }
        }
#pragma omp critical(m_height_highs_max)
        best = std::max(best, local_best);
    }
    if (error) std::rethrow_exception(error);
    return std::min(best, threshold);
}

inline double primal_value(Result result) {
    if (result.status == Status::Optimal) return result.value;
    if (result.status == Status::Unbounded) return std::numeric_limits<double>::infinity();
    throw std::runtime_error("Primal LP unexpectedly infeasible (u=0 is feasible).");
}
} // namespace m_height_lp
#endif
