#include "methods.hh"

#ifdef HAVE_HIGHS
#include <highs/Highs.h> 
#endif // HAVE_HIGHS

#include <glpk.h>
#include <Eigen/Dense>
#include <numeric>
#include <memory>
#include <algorithm>
#include <functional>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <sstream>
#include <mutex>
#include <atomic>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// GLPK is not thread-safe; serialize all GLPK API calls.
static std::mutex glpk_mutex;

struct Task {
    int a;
    std::vector<int> X;
    std::vector<int> Y;
};


static std::tuple<double, Eigen::VectorXd, std::tuple<int, std::vector<int>, std::vector<int>>>
h_m_jiang_simplified_lp_glpk_early_quit_full(const Eigen::MatrixXd &G, int m, double early_quit_threshold)
{
    if (G.rows() == 0 || G.cols() == 0) {
        throw std::runtime_error("Error: Received empty matrix!");
    }
    if (m <= 0 || m > 30) {
        throw std::runtime_error("Error: m out of supported range (1..30).");
    }

    const int k = (int)G.rows();
    const int n = (int)G.cols();

    double bestHeight = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd bestU = Eigen::VectorXd::Zero(k);
    std::tuple<int, std::vector<int>, std::vector<int>> bestParams;

    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::vector<Task> tasks;
#pragma omp parallel
    {
        std::vector<Task> localTasks;

#pragma omp for schedule(dynamic)
        for (int a = 0; a < n; ++a) {
            std::vector<int> rem = allIndices;
            rem.erase(std::remove(rem.begin(), rem.end(), a), rem.end());

            std::vector<std::vector<int>> combos;
            std::function<void(int,int,std::vector<int>&)> combGen =
                [&](int offset, int r, std::vector<int>& tmp) {
                    if (r == 0) { combos.push_back(tmp); return; }
                    for (int i = offset; i <= (int)rem.size() - r; ++i) {
                        tmp.push_back(rem[i]);
                        combGen(i + 1, r - 1, tmp);
                        tmp.pop_back();
                    }
                };
            std::vector<int> tmp;
            combGen(0, m - 1, tmp);

            for (const auto &X : combos) {
                std::vector<int> Y;
                Y.reserve(rem.size() - X.size());
                for (int idx : rem)
                    if (std::find(X.begin(), X.end(), idx) == X.end()) Y.push_back(idx);

                Task t;
                t.a = a;
                t.X = X;
                t.Y = std::move(Y);
                localTasks.push_back(std::move(t));
            }
        }

#pragma omp critical
        tasks.insert(tasks.end(), localTasks.begin(), localTasks.end());
    }

    {
        std::lock_guard<std::mutex> lock(glpk_mutex);
        glp_term_out(GLP_OFF);
    }

    std::atomic<bool> quitEarly{false};
    std::atomic<bool> thresholdReached{false};

    std::atomic<bool> fatal{false};
    std::string fatal_msg;
    std::mutex fatal_mtx;

#pragma omp parallel shared(quitEarly, fatal)
    {
        double localBestHeight = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd localBestU = Eigen::VectorXd::Zero(k);
        std::tuple<int, std::vector<int>, std::vector<int>> localBestParams;

#pragma omp for schedule(dynamic)
        for (int ti = 0; ti < (int)tasks.size(); ++ti) {
            if (quitEarly.load(std::memory_order_relaxed)) continue;
            if (fatal.load(std::memory_order_relaxed)) continue;

            const auto &T = tasks[ti];
            const int a = T.a;
            const auto &X = T.X;
            const auto &Y = T.Y;

            if ((int)X.size() != m - 1) {
                fatal.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(fatal_mtx);
                fatal_msg = "Invariant violated: |X| != m-1.";
                continue;
            }

            const int totalRows = (int)X.size() + 2*(int)Y.size();

            int status = 0;
            double objectiveValue = -std::numeric_limits<double>::infinity();
            Eigen::VectorXd u;

            {
                std::lock_guard<std::mutex> lock(glpk_mutex);

                glp_prob *lp = glp_create_prob();
                if (!lp) {
                    fatal.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lk(fatal_mtx);
                    fatal_msg = "glp_create_prob failed.";
                    continue;
                }

                glp_set_obj_dir(lp, GLP_MAX);
                glp_add_rows(lp, totalRows);
                glp_add_cols(lp, k);

                for (int col = 1; col <= k; ++col)
                    glp_set_col_bnds(lp, col, GLP_FR, 0.0, 0.0);

                for (int ii = 0; ii < k; ++ii)
                    glp_set_obj_coef(lp, ii + 1, G(ii, a));

                int rowIndex = 1;

                for (int jx : X) {
                    std::vector<int> idx(k + 1);
                    std::vector<double> val(k + 1);
                    for (int col = 0; col < k; ++col) {
                        idx[col + 1] = col + 1;
                        val[col + 1] = G(col, jx) - G(col, a);
                    }
                    glp_set_row_bnds(lp, rowIndex, GLP_UP, 0.0, 0.0);
                    glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                    rowIndex++;
                }

                for (int jy : Y) {
                    {
                        std::vector<int> idx(k + 1);
                        std::vector<double> val(k + 1);
                        for (int col = 0; col < k; ++col) {
                            idx[col + 1] = col + 1;
                            val[col + 1] = G(col, jy);
                        }
                        glp_set_row_bnds(lp, rowIndex, GLP_UP, 0.0, 1.0);
                        glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                        rowIndex++;
                    }
                    {
                        std::vector<int> idx(k + 1);
                        std::vector<double> val(k + 1);
                        for (int col = 0; col < k; ++col) {
                            idx[col + 1] = col + 1;
                            val[col + 1] = -G(col, jy);
                        }
                        glp_set_row_bnds(lp, rowIndex, GLP_UP, 0.0, 1.0);
                        glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                        rowIndex++;
                    }
                }

                glp_smcp smcp;
                glp_init_smcp(&smcp);
                smcp.msg_lev = GLP_MSG_OFF;

                glp_simplex(lp, &smcp);
                status = glp_get_status(lp);

                if (status == GLP_OPT || status == GLP_FEAS) {
                    objectiveValue = glp_get_obj_val(lp);
                    u.resize(k);
                    for (int ii = 0; ii < k; ++ii) u[ii] = glp_get_col_prim(lp, ii + 1);
                } else if (status == GLP_UNBND) {
                    objectiveValue = std::numeric_limits<double>::infinity();
                    u = Eigen::VectorXd::Zero(k);
                }

                glp_delete_prob(lp);
            }

            if (fatal.load(std::memory_order_relaxed)) continue;

            bool hitThreshold = objectiveValue > early_quit_threshold;
            bool isInf = objectiveValue == std::numeric_limits<double>::infinity();

            if (objectiveValue > localBestHeight) {
                localBestHeight = objectiveValue;
                if (u.size() == k) localBestU = u;
                localBestParams = std::make_tuple(a, X, Y);
            }
            if (hitThreshold) {
                thresholdReached.store(true, std::memory_order_relaxed);
            }
            if (hitThreshold || isInf) {
                quitEarly.store(true, std::memory_order_relaxed);
            }
        }

#pragma omp critical
        {
            if (localBestHeight > bestHeight) {
                bestHeight = localBestHeight;
                bestU = localBestU;
                bestParams = localBestParams;
            }
        }
    }

    if (fatal.load(std::memory_order_relaxed)) {
        throw std::runtime_error(fatal_msg.empty() ? "Fatal error in OpenMP region." : fatal_msg);
    }

    if (thresholdReached.load(std::memory_order_relaxed)) {
        bestHeight = early_quit_threshold;
    }

    return {bestHeight, bestU, bestParams};
}


#ifdef HAVE_HIGHS
static std::tuple<double, Eigen::VectorXd, std::tuple<int, std::vector<int>, std::vector<int>>>
h_m_jiang_simplified_lp_highs_early_quit_full(const Eigen::MatrixXd& G,
                                   int m,
                                   double early_quit_threshold) {
    if (G.rows() == 0 || G.cols() == 0) throw std::runtime_error("Error: Received empty matrix!");
    if (m <= 0 || m > 30) throw std::runtime_error("Error: m out of supported range (1..30).");

    const int k = (int)G.rows();
    const int n = (int)G.cols();
    if (m > n) throw std::runtime_error("Error: m cannot exceed number of columns.");

    double bestHeight = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd bestU = Eigen::VectorXd::Zero(k);
    std::tuple<int, std::vector<int>, std::vector<int>> bestParams;

    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::atomic<bool> quitEarly{false};
    std::atomic<bool> thresholdReached{false};
    std::atomic<bool> fatal{false};
    std::string fatal_msg;
    std::mutex fatal_mtx;

    const int rowsPerTask = (m - 1) + 2 * (n - m);
    const int nnz = rowsPerTask > 0 ? rowsPerTask * k : 0;

#ifdef _OPENMP
    const int maxThreads = std::max(1, omp_get_max_threads());
#else
    const int maxThreads = 1;
#endif
    const int desiredParallelSolves = 16;
    // Match the original and Roth HiGHS APIs: HiGHS shares a global thread
    // scheduler, so changing its thread count between calls can fail. Parallel
    // work is distributed across independent LPs below.
    const int desiredHighsThreads = 1;
    int highsThreadsPerSolve = std::max(1, std::min(desiredHighsThreads, maxThreads));
    int maxSimultaneousSolves = std::max(1, maxThreads / highsThreadsPerSolve);
    if (maxSimultaneousSolves > desiredParallelSolves) maxSimultaneousSolves = desiredParallelSolves;
    if (maxSimultaneousSolves > 1 && highsThreadsPerSolve > 1) {
        // Avoid nested parallelism: multiple multi-threaded HiGHS solves in
        // parallel trigger passModel failures, so fall back to 1 thread/solve.
        highsThreadsPerSolve = 1;
        maxSimultaneousSolves = std::min(maxThreads, desiredParallelSolves);
    }

    #pragma omp parallel num_threads(maxSimultaneousSolves) shared(quitEarly,fatal,thresholdReached)
    {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("threads", highsThreadsPerSolve);
        // highs.setOptionValue("solver", "simplex");
        // highs.setOptionValue("simplex_strategy", 0);
        highs.setOptionValue("presolve", "off");
        highs.setOptionValue("scaling", "off");

        const double HINF = 1e20;
        highs.setOptionValue("infinite_bound", HINF);
        highs.setOptionValue("infinite_cost",  HINF);

        HighsLp lp;
        if (rowsPerTask > 0) {
            const int xRowCount = m - 1;
            const int yRowCount = n - m;

            lp.num_col_ = k;
            lp.num_row_ = rowsPerTask;
            lp.col_cost_.assign(k, 0.0);
            lp.col_lower_.assign(k, -HINF);
            lp.col_upper_.assign(k,  HINF);
            lp.row_lower_.assign(rowsPerTask, -HINF);
            lp.row_upper_.assign(rowsPerTask, 0.0);
            for (int r = xRowCount; r < xRowCount + yRowCount; ++r) lp.row_upper_[r] = 1.0;
            for (int r = xRowCount + yRowCount; r < rowsPerTask; ++r) lp.row_upper_[r] = 1.0;

            lp.a_matrix_.format_ = MatrixFormat::kRowwise;
            lp.a_matrix_.start_.resize(rowsPerTask + 1);
            for (int r = 0; r <= rowsPerTask; ++r) lp.a_matrix_.start_[r] = r * k;
            lp.a_matrix_.index_.resize(nnz);
            for (int r = 0; r < rowsPerTask; ++r)
                for (int col = 0; col < k; ++col)
                    lp.a_matrix_.index_[r * k + col] = col;
            lp.a_matrix_.value_.assign(nnz, 0.0);
            lp.sense_ = ObjSense::kMaximize;
        }

        std::vector<int> currentX;
        currentX.reserve(m);
        std::vector<int> Ybuffer;
        Ybuffer.reserve(n);

        double localBestHeight = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd localBestU = Eigen::VectorXd::Zero(k);
        std::tuple<int, std::vector<int>, std::vector<int>> localBestParams;

#pragma omp for schedule(dynamic)
        for (int a = 0; a < n; ++a) {
            if (quitEarly.load(std::memory_order_acquire)) {
                #if defined(_OPENMP) && _OPENMP >= 201307
                #pragma omp cancel for
                #endif
                continue;
            }
            if (fatal.load(std::memory_order_acquire)) continue;

            std::vector<int> rem = allIndices;
            rem.erase(std::remove(rem.begin(), rem.end(), a), rem.end());

            if ((int)rem.size() != n - 1) {
                fatal.store(true, std::memory_order_release);
                std::lock_guard<std::mutex> lk(fatal_mtx);
                fatal_msg = "Invariant violated: rem size mismatch.";
                continue;
            }

            if (rowsPerTask == 0) {
                double maxCost = 0.0;
                for (int ii = 0; ii < k; ++ii) maxCost = std::max(maxCost, std::abs(G(ii, a)));

                double objectiveValue = 0.0;
                Eigen::VectorXd u = Eigen::VectorXd::Zero(k);
                if (maxCost > 0.0) objectiveValue = std::numeric_limits<double>::infinity();

                bool hitThreshold = objectiveValue > early_quit_threshold;
                bool isInf = objectiveValue == std::numeric_limits<double>::infinity();

                if (objectiveValue > localBestHeight) {
                    localBestHeight = objectiveValue;
                    localBestU = u;
                    localBestParams = std::make_tuple(a, std::vector<int>{}, std::vector<int>{});
                }
                if (hitThreshold) {
                    thresholdReached.store(true, std::memory_order_release);
                }
                if (hitThreshold || isInf) {
                    quitEarly.store(true, std::memory_order_release);
                    #if defined(_OPENMP) && _OPENMP >= 201307
                    #pragma omp cancel for
                    #endif
                }
                #if defined(_OPENMP) && _OPENMP >= 201307
                #pragma omp cancellation point for
                #endif
                continue;
            }

            for (int ii = 0; ii < k; ++ii) lp.col_cost_[ii] = G(ii, a);

            currentX.clear();

            std::function<void(int,int)> dfs = [&](int offset, int need) {
                if (quitEarly.load(std::memory_order_acquire)) return;
                if (fatal.load(std::memory_order_acquire)) return;
                if (need == 0) {
                    Ybuffer.clear();
                    Ybuffer.reserve(rem.size() - currentX.size());
                    auto xIt = currentX.begin();
                    for (int idx : rem) {
                        if (xIt != currentX.end() && *xIt == idx) {
                            ++xIt;
                        } else {
                            Ybuffer.push_back(idx);
                        }
                    }

                    if ((int)Ybuffer.size() != n - m) {
                        fatal.store(true, std::memory_order_release);
                        std::lock_guard<std::mutex> lk(fatal_mtx);
                        fatal_msg = "Invariant violated: |Y| != n-m.";
                        return;
                    }

                    if (!lp.a_matrix_.value_.empty()) {
                        std::fill(lp.a_matrix_.value_.begin(), lp.a_matrix_.value_.end(), 0.0);
                    }

                    int rowIndex = 0;
                    for (int jx : currentX) {
                        const int base = rowIndex * k;
                        for (int col = 0; col < k; ++col) {
                            lp.a_matrix_.value_[base + col] = G(col, jx) - G(col, a);
                        }
                        rowIndex++;
                    }
                    for (int jy : Ybuffer) {
                        const int basePos = rowIndex * k;
                        for (int col = 0; col < k; ++col) {
                            lp.a_matrix_.value_[basePos + col] = G(col, jy);
                        }
                        rowIndex++;
                        const int baseNeg = rowIndex * k;
                        for (int col = 0; col < k; ++col) {
                            lp.a_matrix_.value_[baseNeg + col] = -G(col, jy);
                        }
                        rowIndex++;
                    }

                    if (rowIndex != rowsPerTask) {
                        fatal.store(true, std::memory_order_release);
                        std::lock_guard<std::mutex> lk(fatal_mtx);
                        fatal_msg = "HiGHS: row count mismatch before passModel.";
                        return;
                    }

                    double objectiveValue = -std::numeric_limits<double>::infinity();
                    Eigen::VectorXd u;

                    highs.clearModel();
                    highs.clearSolver();

                    if (highs.passModel(lp) != HighsStatus::kOk) {
                        fatal.store(true, std::memory_order_release);
                        std::lock_guard<std::mutex> lk(fatal_mtx);
                        std::ostringstream oss;
                        oss << "HiGHS: passModel failed (a=" << a
                            << ", |X|=" << currentX.size()
                            << ", |Y|=" << Ybuffer.size()
                            << ", m=" << m << ", n=" << n << ", X=[";
                        for (size_t idx = 0; idx < currentX.size(); ++idx) {
                            if (idx) oss << ",";
                            oss << currentX[idx];
                        }
                        oss << "], Y=[";
                        for (size_t idx = 0; idx < Ybuffer.size(); ++idx) {
                            if (idx) oss << ",";
                            oss << Ybuffer[idx];
                            if (idx >= 5 && idx + 1 < Ybuffer.size()) {
                                oss << ",...";
                                break;
                            }
                        }
                        oss << "])";
                        fatal_msg = oss.str();
                        return;
                    }
                    if (highs.run() != HighsStatus::kOk) {
                        fatal.store(true, std::memory_order_release);
                        std::lock_guard<std::mutex> lk(fatal_mtx);
                        std::ostringstream oss;
                        oss << "HiGHS: run() failed (a=" << a
                            << ", |X|=" << currentX.size()
                            << ", |Y|=" << Ybuffer.size()
                            << ", m=" << m << ", n=" << n << ", X=[";
                        for (size_t idx = 0; idx < currentX.size(); ++idx) {
                            if (idx) oss << ",";
                            oss << currentX[idx];
                        }
                        oss << "], Y=[";
                        for (size_t idx = 0; idx < Ybuffer.size(); ++idx) {
                            if (idx) oss << ",";
                            oss << Ybuffer[idx];
                            if (idx >= 5 && idx + 1 < Ybuffer.size()) {
                                oss << ",...";
                                break;
                            }
                        }
                        oss << "])";
                        fatal_msg = oss.str();
                        return;
                    }

                    const HighsModelStatus ms = highs.getModelStatus();

                    if (ms == HighsModelStatus::kUnbounded || ms == HighsModelStatus::kUnboundedOrInfeasible) {
                        objectiveValue = std::numeric_limits<double>::infinity();
                        u = Eigen::VectorXd::Zero(k);
                    } else if (ms == HighsModelStatus::kOptimal || ms == HighsModelStatus::kObjectiveBound || ms == HighsModelStatus::kNotset) {
                        objectiveValue = highs.getInfo().objective_function_value;
                        const auto& sol = highs.getSolution();
                        u.resize(k);
                        for (int i = 0; i < k; ++i) u[i] = sol.col_value[i];
                    }

                    bool hitThreshold = objectiveValue > early_quit_threshold;
                    bool isInf = objectiveValue == std::numeric_limits<double>::infinity();

                    if (objectiveValue > localBestHeight) {
                        localBestHeight = objectiveValue;
                        if (u.size() == k) localBestU = u;
                        localBestParams = std::make_tuple(a, currentX, Ybuffer);
                    }
                    if (hitThreshold) {
                        thresholdReached.store(true, std::memory_order_release);
                    }
                    if (hitThreshold || isInf) {
                        quitEarly.store(true, std::memory_order_release);
                    }
                    return;
                }

                for (int i = offset; i <= (int)rem.size() - need; ++i) {
                    currentX.push_back(rem[i]);
                    dfs(i + 1, need - 1);
                    currentX.pop_back();
                    if (quitEarly.load(std::memory_order_acquire)) return;
                    if (fatal.load(std::memory_order_acquire)) return;
                }
            };

            if (m - 1 == 0) {
                dfs(0, 0);
            } else {
                dfs(0, m - 1);
            }

            if (quitEarly.load(std::memory_order_acquire)) {
                #if defined(_OPENMP) && _OPENMP >= 201307
                #pragma omp cancel for
                #endif
            }
            #if defined(_OPENMP) && _OPENMP >= 201307
            #pragma omp cancellation point for
            #endif
        }

        #pragma omp critical
        {
            if (localBestHeight > bestHeight) {
                bestHeight = localBestHeight;
                bestU = localBestU;
                bestParams = localBestParams;
            }
        }
    }

    if (fatal.load(std::memory_order_acquire)) {
        if (!fatal_msg.empty() && fatal_msg.rfind("HiGHS:", 0) == 0) {
            auto glpk_res = h_m_jiang_simplified_lp_glpk_early_quit_full(G, m, early_quit_threshold);
            double& fallback_height = std::get<0>(glpk_res);
            if (fallback_height > early_quit_threshold) {
                fallback_height = early_quit_threshold;
            }
            return glpk_res;
        }
        throw std::runtime_error(fatal_msg.empty() ? "Fatal error in OpenMP region (HiGHS early quit)." : fatal_msg);
    }

    if (thresholdReached.load(std::memory_order_acquire)) {
        bestHeight = early_quit_threshold;
    }

    return {bestHeight, bestU, bestParams};
}
#endif // HAVE_HIGHS

double h_m_jiang_simplified_lp_glpk_early_quit(const Eigen::MatrixXd& G,
                                    int m,
                                    double early_quit_threshold) {
    return std::get<0>(h_m_jiang_simplified_lp_glpk_early_quit_full(G, m, early_quit_threshold));
}

#ifdef HAVE_HIGHS
double h_m_jiang_simplified_lp_highs_early_quit(const Eigen::MatrixXd& G,
                                     int m,
                                     double early_quit_threshold) {
    return std::get<0>(h_m_jiang_simplified_lp_highs_early_quit_full(G, m, early_quit_threshold));
}
#endif


namespace {
enum class OriginalBackend { Glpk, Highs };

// Both backends receive identical coefficients. Only equality_row is an
// equality (c_b = 1); every other constraint is an upper bound.
double solve_original_lp(const Eigen::MatrixXd& A, const Eigen::VectorXd& bounds,
                         const Eigen::VectorXd& cost, int equality_row,
                         OriginalBackend backend) {
    const int rows = A.rows(), k = A.cols();
    if (backend == OriginalBackend::Glpk) {
        std::lock_guard<std::mutex> lock(glpk_mutex);
        std::unique_ptr<glp_prob, decltype(&glp_delete_prob)> lp(
            glp_create_prob(), &glp_delete_prob);
        if (!lp) throw std::runtime_error("GLPK original LP: allocation failed.");
        glp_set_obj_dir(lp.get(), GLP_MAX);
        glp_add_rows(lp.get(), rows);
        glp_add_cols(lp.get(), k);
        for (int j = 0; j < k; ++j) {
            glp_set_col_bnds(lp.get(), j + 1, GLP_FR, 0.0, 0.0);
            glp_set_obj_coef(lp.get(), j + 1, cost[j]);
        }
        std::vector<int> indices(k + 1);
        std::vector<double> values(k + 1);
        for (int j = 0; j < k; ++j) indices[j + 1] = j + 1;
        for (int row = 0; row < rows; ++row) {
            glp_set_row_bnds(lp.get(), row + 1,
                            row == equality_row ? GLP_FX : GLP_UP,
                            bounds[row], bounds[row]);
            for (int j = 0; j < k; ++j) values[j + 1] = A(row, j);
            glp_set_mat_row(lp.get(), row + 1, k, indices.data(), values.data());
        }
        glp_smcp options;
        glp_init_smcp(&options);
        options.msg_lev = GLP_MSG_OFF;
        const int result = glp_simplex(lp.get(), &options);
        if (result != 0)
            throw std::runtime_error("GLPK original LP: simplex failed: " + std::to_string(result));
        const int status = glp_get_status(lp.get());
        if (status == GLP_OPT) return glp_get_obj_val(lp.get());
        if (status == GLP_NOFEAS) return -std::numeric_limits<double>::infinity();
        if (status == GLP_UNBND) return std::numeric_limits<double>::infinity();
        throw std::runtime_error("GLPK original LP: unexpected status: " + std::to_string(status));
    }
#ifdef HAVE_HIGHS
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("threads", 1);
    highs.setOptionValue("solver", "simplex");
    highs.setOptionValue("simplex_strategy", 0);
    highs.setOptionValue("presolve", "off");
    highs.setOptionValue("scaling", "off");
    constexpr double inf = 1e20;
    highs.setOptionValue("infinite_bound", inf);
    highs.setOptionValue("infinite_cost", inf);
    HighsLp lp;
    lp.num_col_ = k;
    lp.num_row_ = rows;
    lp.sense_ = ObjSense::kMaximize;
    lp.col_cost_.assign(cost.data(), cost.data() + k);
    lp.col_lower_.assign(k, -inf);
    lp.col_upper_.assign(k, inf);
    lp.row_lower_.assign(rows, -inf);
    lp.row_lower_[equality_row] = bounds[equality_row];
    lp.row_upper_.assign(bounds.data(), bounds.data() + rows);
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.start_.resize(rows + 1);
    lp.a_matrix_.index_.resize(rows * k);
    lp.a_matrix_.value_.resize(rows * k);
    for (int row = 0; row <= rows; ++row) lp.a_matrix_.start_[row] = row * k;
    for (int row = 0; row < rows; ++row)
        for (int j = 0; j < k; ++j) {
            lp.a_matrix_.index_[row * k + j] = j;
            lp.a_matrix_.value_[row * k + j] = A(row, j);
        }
    if (highs.passModel(lp) != HighsStatus::kOk || highs.run() != HighsStatus::kOk)
        throw std::runtime_error("HiGHS original LP: solve failed.");
    const auto status = highs.getModelStatus();
    if (status == HighsModelStatus::kOptimal) return highs.getInfo().objective_function_value;
    if (status == HighsModelStatus::kInfeasible) return -std::numeric_limits<double>::infinity();
    if (status == HighsModelStatus::kUnbounded) return std::numeric_limits<double>::infinity();
    throw std::runtime_error("HiGHS original LP: unexpected status: " + highs.modelStatusToString(status));
#else
    throw std::runtime_error("Jiang original HiGHS LP requires HAVE_HIGHS.");
#endif
}

double jiang_original_lp(const Eigen::MatrixXd& G, int m, OriginalBackend backend) {
    const int k = G.rows(), n = G.cols();
    if (k == 0 || n < 2 || !G.allFinite())
        throw std::invalid_argument("G must be finite, nonempty, with at least two columns.");
    if (m < 1 || m >= n || m > 30)
        throw std::invalid_argument("Jiang original LP requires 1 <= m <= min(30, n-1).");
    for (int j = 0; j < n; ++j)
        if (G.col(j).isZero(0.0))
            throw std::invalid_argument("Jiang original LP assumes no zero column.");

    const int rows = 2 * (m - 1) + 1 + 2 * (n - m - 1);
    const int equality_row = 2 * (m - 1);
    Eigen::MatrixXd A(rows, k);
    Eigen::VectorXd bounds(rows), cost(k);
    double best = 0.0;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            std::vector<int> remaining, X;
            for (int j = 0; j < n; ++j)
                if (j != a && j != b) remaining.push_back(j);
            std::function<void(int, int)> enumerate = [&](int start, int need) {
                if (need != 0) {
                    for (int t = start; t <= static_cast<int>(remaining.size()) - need; ++t) {
                        X.push_back(remaining[t]);
                        enumerate(t + 1, need - 1);
                        X.pop_back();
                    }
                    return;
                }
                std::vector<int> Y;
                for (int j : remaining)
                    if (!std::binary_search(X.begin(), X.end(), j)) Y.push_back(j);
                for (unsigned long long mask = 0; mask < (1ULL << m); ++mask) {
                    const double sa = (mask & 1ULL) ? 1.0 : -1.0;
                    cost = sa * G.col(a);
                    int row = 0;
                    for (int t = 0; t < static_cast<int>(X.size()); ++t) {
                        const double sx = (mask & (1ULL << (t + 1))) ? 1.0 : -1.0;
                        // 1 <= sx*c_x <= sa*c_a.
                        A.row(row) = (sx * G.col(X[t]) - cost).transpose();
                        bounds[row++] = 0.0;
                        A.row(row) = (-sx * G.col(X[t])).transpose();
                        bounds[row++] = -1.0;
                    }
                    A.row(row) = G.col(b).transpose();
                    bounds[row++] = 1.0;
                    for (int j : Y) {
                        A.row(row) = G.col(j).transpose();
                        bounds[row++] = 1.0;
                        A.row(row) = -G.col(j).transpose();
                        bounds[row++] = 1.0;
                    }
                    best = std::max(best, solve_original_lp(A, bounds, cost, equality_row, backend));
                }
            };
            enumerate(0, m - 1);
        }
    }
    return best;
}
} // namespace

double h_m_jiang_original_lp_glpk(const Eigen::MatrixXd& G, int m) {
    return jiang_original_lp(G, m, OriginalBackend::Glpk);
}
#ifdef HAVE_HIGHS
double h_m_jiang_original_lp_highs(const Eigen::MatrixXd& G, int m) {
    return jiang_original_lp(G, m, OriginalBackend::Highs);
}
#endif
