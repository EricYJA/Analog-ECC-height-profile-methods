#include "methods.hh"

#ifdef HAVE_HIGHS
#include <highs/Highs.h> 
#endif // HAVE_HIGHS

#include <glpk.h>
#include <Eigen/Dense>
#include <numeric>
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
h_m_jiang_lp_glpk_full(const Eigen::MatrixXd &G, int m)
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

    // Precompute all tasks in parallel, then merge.
    std::vector<Task> tasks;
#pragma omp parallel
    {
        std::vector<Task> localTasks;

#pragma omp for schedule(dynamic)
        for (int a = 0; a < n; ++a) {
            std::vector<int> rem = allIndices;
            rem.erase(std::remove(rem.begin(), rem.end(), a), rem.end());

            // combinations of size (m-1) from rem
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

    // Disable GLPK terminal output once.
    {
        std::lock_guard<std::mutex> lock(glpk_mutex);
        glp_term_out(GLP_OFF);
    }

    std::atomic<bool> quitEarly{false};

    // Record fatal errors without throwing inside the parallel region.
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

            // |X| must be m-1 (construction guarantees this, but keep a guard)
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

            // Serialize all GLPK API calls.
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
                    glp_set_col_bnds(lp, col, GLP_FR, 0.0, 0.0); // free vars

                for (int ii = 0; ii < k; ++ii)
                    glp_set_obj_coef(lp, ii + 1, G(ii, a));

                int rowIndex = 1;

                // X rows: (G(:,jx) - G(:,a)) · u <= 0
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

                // Y rows: G(:,jy)·u ≤ 1 and -G(:,jy)·u ≤ 1
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
                smcp.msg_lev = GLP_MSG_OFF;               // silence simplex and presolver
                // smcp.presolve = GLP_ON;                // enable presolver (default)

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

            if (objectiveValue > localBestHeight) {
                localBestHeight = objectiveValue;
                if (u.size() == k) localBestU = u;
                localBestParams = std::make_tuple(a, X, Y);
            }
            if (objectiveValue == std::numeric_limits<double>::infinity()) {
                quitEarly.store(true, std::memory_order_relaxed);
            }
        } // tasks loop

#pragma omp critical
        {
            if (localBestHeight > bestHeight) {
                bestHeight = localBestHeight;
                bestU = localBestU;
                bestParams = localBestParams;
            }
        }
    } // parallel region

    if (fatal.load(std::memory_order_relaxed)) {
        throw std::runtime_error(fatal_msg.empty() ? "Fatal error in OpenMP region." : fatal_msg);
    }

    return {bestHeight, bestU, bestParams};
}

static std::tuple<double, Eigen::VectorXd, std::tuple<int, std::vector<int>, std::vector<int>>>
h_m_jiang_lp_glpk_early_quit_full(const Eigen::MatrixXd &G, int m, double early_quit_threshold)
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
h_m_jiang_lp_highs_full_impl(const Eigen::MatrixXd& G, int m, bool add_more_constraint) {
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
    std::atomic<bool> fatal{false};
    std::string fatal_msg;
    std::mutex fatal_mtx;

    const int xConstraintRows = (add_more_constraint ? 2 : 1) * (m - 1);
    const int rowsPerTask = xConstraintRows + 2 * (n - m);
    const int nnz = rowsPerTask > 0 ? rowsPerTask * k : 0;

#ifdef _OPENMP
    // const int maxThreads = std::max(1, omp_get_max_threads());
#else
    // const int maxThreads = 1;
#endif
    const int maxThreads = 1; // TODO: Temporarily disable parallelism

    const int desiredParallelSolves = 16;
    const int desiredHighsThreads = 2;
    const int highsThreadsPerSolve = std::min(desiredHighsThreads, maxThreads);
    int maxSimultaneousSolves = std::max(1, maxThreads / highsThreadsPerSolve);
    if (maxSimultaneousSolves > desiredParallelSolves) maxSimultaneousSolves = desiredParallelSolves;

    #pragma omp parallel num_threads(maxSimultaneousSolves) shared(quitEarly,fatal)
    {
        // Thread-local HiGHS instance and scratch buffers
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("threads", highsThreadsPerSolve);   // use up to 4 threads per solve
        highs.setOptionValue("solver", "simplex");              // cheaper than IPM here
        highs.setOptionValue("simplex_strategy", 0);            // dual simplex, standard
        highs.setOptionValue("presolve", "off");                // cut startup work
        highs.setOptionValue("scaling", "off");                 // cut startup work
        // highs.setOptionValue("time_limit", 0.2);             // optional per-LP cap

        const double HINF = 1e20;                               // sane large bound
        highs.setOptionValue("infinite_bound", HINF);
        highs.setOptionValue("infinite_cost",  HINF);

        HighsLp lp;
        if (rowsPerTask > 0) {
            const int yRowCount = n - m;

            lp.num_col_ = k;
            lp.num_row_ = rowsPerTask;
            lp.col_cost_.assign(k, 0.0);
            lp.col_lower_.assign(k, -HINF);
            lp.col_upper_.assign(k,  HINF);
            lp.row_lower_.assign(rowsPerTask, -HINF);
            lp.row_upper_.assign(rowsPerTask, 0.0);
            for (int r = xConstraintRows; r < xConstraintRows + yRowCount; ++r) lp.row_upper_[r] = 1.0;
            for (int r = xConstraintRows + yRowCount; r < rowsPerTask; ++r) lp.row_upper_[r] = 1.0;

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

                if (objectiveValue > localBestHeight) {
                    localBestHeight = objectiveValue;
                    localBestU = u;
                    localBestParams = std::make_tuple(a, std::vector<int>{}, std::vector<int>{});
                }
                if (objectiveValue == std::numeric_limits<double>::infinity()) {
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
                        if (add_more_constraint) {
                            const int moreBase = rowIndex * k;
                            for (int col = 0; col < k; ++col) {
                                lp.a_matrix_.value_[moreBase + col] = -G(col, jx) - G(col, a);
                            }
                            rowIndex++;
                        }
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

                    if (objectiveValue > localBestHeight) {
                        localBestHeight = objectiveValue;
                        if (u.size() == k) localBestU = u;
                        localBestParams = std::make_tuple(a, currentX, Ybuffer);
                    }
                    if (objectiveValue == std::numeric_limits<double>::infinity()) {
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
        if (!add_more_constraint && !fatal_msg.empty() && fatal_msg.rfind("HiGHS:", 0) == 0) {
            return h_m_jiang_lp_glpk_full(G, m);
        }
        throw std::runtime_error(fatal_msg.empty() ? "Fatal error in OpenMP region (HiGHS)." : fatal_msg);
    }
    return {bestHeight, bestU, bestParams};
}

static std::tuple<double, Eigen::VectorXd, std::tuple<int, std::vector<int>, std::vector<int>>>
h_m_jiang_lp_highs_full(const Eigen::MatrixXd& G, int m) {
    return h_m_jiang_lp_highs_full_impl(G, m, false);
}

static std::tuple<double, Eigen::VectorXd, std::tuple<int, std::vector<int>, std::vector<int>>>
h_m_jiang_lp_highs_more_constraint_full(const Eigen::MatrixXd& G, int m) {
    return h_m_jiang_lp_highs_full_impl(G, m, true);
}

static std::tuple<double, Eigen::VectorXd, std::tuple<int, std::vector<int>, std::vector<int>>>
h_m_jiang_lp_highs_early_quit_full(const Eigen::MatrixXd& G,
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
    const int desiredHighsThreads = 2;
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
            auto glpk_res = h_m_jiang_lp_glpk_full(G, m);
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

double h_m_jiang_lp_glpk(const Eigen::MatrixXd& G, int m) {
    return std::get<0>(h_m_jiang_lp_glpk_full(G, m));
}

double h_m_jiang_lp_glpk_early_quit(const Eigen::MatrixXd& G,
                                    int m,
                                    double early_quit_threshold) {
    return std::get<0>(h_m_jiang_lp_glpk_early_quit_full(G, m, early_quit_threshold));
}

#ifdef HAVE_HIGHS
double h_m_jiang_lp_highs(const Eigen::MatrixXd& G, int m) {
    return std::get<0>(h_m_jiang_lp_highs_full(G, m));
}

double h_m_jiang_lp_highs_more_constraint(const Eigen::MatrixXd& G, int m) {
    return std::get<0>(h_m_jiang_lp_highs_more_constraint_full(G, m));
}

double h_m_jiang_lp_highs_early_quit(const Eigen::MatrixXd& G,
                                     int m,
                                     double early_quit_threshold) {
    return std::get<0>(h_m_jiang_lp_highs_early_quit_full(G, m, early_quit_threshold));
}
#endif

#ifdef HAVE_HIGHS

struct JiangOriginalParams {
    int a = -1;
    int b = -1;
    std::vector<int> X;
    std::vector<int> Y;
    std::vector<int> psi;  // length m, entries in {-1, +1}
};

static std::tuple<double, Eigen::VectorXd, JiangOriginalParams>
h_m_jiang_original_highs_full_impl(const Eigen::MatrixXd& G, int m)
{
    if (G.rows() == 0 || G.cols() == 0) {
        throw std::runtime_error("Error: Received empty matrix!");
    }
    if (m <= 0 || m > 30) {
        throw std::runtime_error("Error: m out of supported range (1..30).");
    }

    const int k = (int)G.rows();
    const int n = (int)G.cols();

    if (n < 2) {
        throw std::runtime_error("Error: n must be at least 2.");
    }
    if (m >= n) {
        throw std::runtime_error("Error: Jiang original LP requires 1 <= m <= n-1.");
    }

    for (int j = 0; j < n; ++j) {
        if (G.col(j).squaredNorm() == 0.0) {
            std::ostringstream oss;
            oss << "Error: Jiang original LP assumes no zero column; column "
                << j << " is zero.";
            throw std::runtime_error(oss.str());
        }
    }

    double bestHeight = 0.0;
    Eigen::VectorXd bestU = Eigen::VectorXd::Zero(k);
    JiangOriginalParams bestParams;

    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::atomic<bool> fatal{false};
    std::string fatal_msg;
    std::mutex fatal_mtx;

    auto recordFatal = [&](const std::string& msg) {
        if (!fatal.exchange(true, std::memory_order_acq_rel)) {
            std::lock_guard<std::mutex> lk(fatal_mtx);
            fatal_msg = msg;
        }
    };

    const int xCount = m - 1;
    const int yCount = n - m - 1;
    const int rowsPerTask = 2 * xCount + 1 + 2 * yCount;
    const int nnz = rowsPerTask * k;

    const int maxThreads = 1;

    const int desiredParallelSolves = 16;
    const int highsThreadsPerSolve = 1;
    const int maxSimultaneousSolves = std::max(1, std::min(maxThreads, desiredParallelSolves));

#pragma omp parallel num_threads(maxSimultaneousSolves) shared(fatal)
    {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("threads", highsThreadsPerSolve);
        highs.setOptionValue("solver", "simplex");
        highs.setOptionValue("simplex_strategy", 0);
        highs.setOptionValue("presolve", "off");
        highs.setOptionValue("scaling", "off");

        const double HINF = 1e20;
        highs.setOptionValue("infinite_bound", HINF);
        highs.setOptionValue("infinite_cost",  HINF);

        HighsLp lp;
        lp.num_col_ = k;
        lp.num_row_ = rowsPerTask;
        lp.col_cost_.assign(k, 0.0);
        lp.col_lower_.assign(k, -HINF);
        lp.col_upper_.assign(k,  HINF);
        lp.row_lower_.assign(rowsPerTask, -HINF);
        lp.row_upper_.assign(rowsPerTask, 0.0);
        lp.sense_ = ObjSense::kMaximize;

        int rowIndexForBounds = 0;

        for (int t = 0; t < xCount; ++t) {
            lp.row_upper_[rowIndexForBounds] = 0.0;
            rowIndexForBounds++;

            lp.row_upper_[rowIndexForBounds] = -1.0;
            rowIndexForBounds++;
        }

        const int equalityRow = rowIndexForBounds;
        lp.row_lower_[equalityRow] = 1.0;
        lp.row_upper_[equalityRow] = 1.0;
        rowIndexForBounds++;

        for (int t = 0; t < yCount; ++t) {
            lp.row_upper_[rowIndexForBounds] = 1.0;
            rowIndexForBounds++;

            lp.row_upper_[rowIndexForBounds] = 1.0;
            rowIndexForBounds++;
        }

        if (rowIndexForBounds != rowsPerTask) {
            recordFatal("HiGHS original LP: row-bound count mismatch.");
        }

        lp.a_matrix_.format_ = MatrixFormat::kRowwise;
        lp.a_matrix_.start_.resize(rowsPerTask + 1);
        for (int r = 0; r <= rowsPerTask; ++r) {
            lp.a_matrix_.start_[r] = r * k;
        }

        lp.a_matrix_.index_.resize(nnz);
        for (int r = 0; r < rowsPerTask; ++r) {
            for (int col = 0; col < k; ++col) {
                lp.a_matrix_.index_[r * k + col] = col;
            }
        }
        lp.a_matrix_.value_.assign(nnz, 0.0);

        std::vector<int> rem;
        rem.reserve(n);

        std::vector<int> currentX;
        currentX.reserve(std::max(0, xCount));

        std::vector<int> Ybuffer;
        Ybuffer.reserve(std::max(0, yCount));

        std::vector<int> psi(m, 1);

        double localBestHeight = 0.0;
        Eigen::VectorXd localBestU = Eigen::VectorXd::Zero(k);
        JiangOriginalParams localBestParams;

        auto solveOneLP = [&](int a,
                              int b,
                              const std::vector<int>& X,
                              const std::vector<int>& Y,
                              const std::vector<int>& currentPsi) {
            if (fatal.load(std::memory_order_acquire)) return;

            const int s0 = currentPsi[0];

            for (int col = 0; col < k; ++col) {
                lp.col_cost_[col] = s0 * G(col, a);
            }

            std::fill(lp.a_matrix_.value_.begin(), lp.a_matrix_.value_.end(), 0.0);

            int row = 0;

            for (int xpos = 0; xpos < (int)X.size(); ++xpos) {
                const int x = X[xpos];
                const int sx = currentPsi[xpos + 1];

                {
                    const int base = row * k;
                    for (int col = 0; col < k; ++col) {
                        lp.a_matrix_.value_[base + col] = sx * G(col, x) - s0 * G(col, a);
                    }
                    row++;
                }

                {
                    const int base = row * k;
                    for (int col = 0; col < k; ++col) {
                        lp.a_matrix_.value_[base + col] = -sx * G(col, x);
                    }
                    row++;
                }
            }

            {
                const int base = row * k;
                for (int col = 0; col < k; ++col) {
                    lp.a_matrix_.value_[base + col] = G(col, b);
                }
                row++;
            }

            for (int y : Y) {
                {
                    const int base = row * k;
                    for (int col = 0; col < k; ++col) {
                        lp.a_matrix_.value_[base + col] = G(col, y);
                    }
                    row++;
                }

                {
                    const int base = row * k;
                    for (int col = 0; col < k; ++col) {
                        lp.a_matrix_.value_[base + col] = -G(col, y);
                    }
                    row++;
                }
            }

            if (row != rowsPerTask) {
                std::ostringstream oss;
                oss << "HiGHS original LP: row count mismatch before passModel "
                    << "(a=" << a << ", b=" << b
                    << ", |X|=" << X.size()
                    << ", |Y|=" << Y.size()
                    << ", m=" << m << ", n=" << n << ").";
                recordFatal(oss.str());
                return;
            }

            highs.clearModel();
            highs.clearSolver();

            if (highs.passModel(lp) != HighsStatus::kOk) {
                std::ostringstream oss;
                oss << "HiGHS original LP: passModel failed "
                    << "(a=" << a << ", b=" << b
                    << ", |X|=" << X.size()
                    << ", |Y|=" << Y.size()
                    << ", m=" << m << ", n=" << n << ").";
                recordFatal(oss.str());
                return;
            }

            if (highs.run() != HighsStatus::kOk) {
                std::ostringstream oss;
                oss << "HiGHS original LP: run() failed "
                    << "(a=" << a << ", b=" << b
                    << ", |X|=" << X.size()
                    << ", |Y|=" << Y.size()
                    << ", m=" << m << ", n=" << n << ").";
                recordFatal(oss.str());
                return;
            }

            double objectiveValue = 0.0;
            Eigen::VectorXd u = Eigen::VectorXd::Zero(k);

            const HighsModelStatus ms = highs.getModelStatus();

            if (ms == HighsModelStatus::kUnbounded ||
                ms == HighsModelStatus::kUnboundedOrInfeasible) {
                objectiveValue = std::numeric_limits<double>::infinity();
            } else if (ms == HighsModelStatus::kInfeasible) {
                objectiveValue = 0.0;
            } else if (ms == HighsModelStatus::kOptimal ||
                       ms == HighsModelStatus::kObjectiveBound) {
                objectiveValue = highs.getInfo().objective_function_value;
                const auto& sol = highs.getSolution();
                u.resize(k);
                for (int i = 0; i < k; ++i) {
                    u[i] = sol.col_value[i];
                }
            } else {
                std::ostringstream oss;
                oss << "HiGHS original LP: unexpected model status "
                    << static_cast<int>(ms)
                    << " (a=" << a << ", b=" << b
                    << ", |X|=" << X.size()
                    << ", |Y|=" << Y.size()
                    << ", m=" << m << ", n=" << n << ").";
                recordFatal(oss.str());
                return;
            }

            if (objectiveValue > localBestHeight) {
                localBestHeight = objectiveValue;
                if (u.size() == k) {
                    localBestU = u;
                }

                JiangOriginalParams p;
                p.a = a;
                p.b = b;
                p.X = X;
                p.Y = Y;
                p.psi = currentPsi;
                localBestParams = std::move(p);
            }
        };

#pragma omp for schedule(dynamic)
        for (int a = 0; a < n; ++a) {
            if (fatal.load(std::memory_order_acquire)) continue;

            for (int b = 0; b < n; ++b) {
                if (b == a) continue;
                if (fatal.load(std::memory_order_acquire)) break;

                rem.clear();
                for (int idx : allIndices) {
                    if (idx != a && idx != b) {
                        rem.push_back(idx);
                    }
                }

                if ((int)rem.size() != n - 2) {
                    std::ostringstream oss;
                    oss << "HiGHS original LP: rem size mismatch "
                        << "(a=" << a << ", b=" << b << ").";
                    recordFatal(oss.str());
                    break;
                }

                currentX.clear();

                std::function<void(int,int)> dfs = [&](int offset, int need) {
                    if (fatal.load(std::memory_order_acquire)) return;

                    if (need == 0) {
                        Ybuffer.clear();
                        auto xIt = currentX.begin();

                        for (int idx : rem) {
                            if (xIt != currentX.end() && *xIt == idx) {
                                ++xIt;
                            } else {
                                Ybuffer.push_back(idx);
                            }
                        }

                        if ((int)Ybuffer.size() != yCount) {
                            std::ostringstream oss;
                            oss << "HiGHS original LP: |Y| mismatch "
                                << "(a=" << a << ", b=" << b
                                << ", |X|=" << currentX.size()
                                << ", |Y|=" << Ybuffer.size()
                                << ", expected |Y|=" << yCount << ").";
                            recordFatal(oss.str());
                            return;
                        }

                        const unsigned long long maskLimit = 1ULL << m;

                        for (unsigned long long mask = 0; mask < maskLimit; ++mask) {
                            if (fatal.load(std::memory_order_acquire)) return;

                            for (int j = 0; j < m; ++j) {
                                psi[j] = ((mask >> j) & 1ULL) ? 1 : -1;
                            }

                            solveOneLP(a, b, currentX, Ybuffer, psi);
                        }

                        return;
                    }

                    for (int i = offset; i <= (int)rem.size() - need; ++i) {
                        currentX.push_back(rem[i]);
                        dfs(i + 1, need - 1);
                        currentX.pop_back();

                        if (fatal.load(std::memory_order_acquire)) return;
                    }
                };

                dfs(0, xCount);
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

    if (fatal.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            fatal_msg.empty()
                ? "Fatal error in OpenMP region for Jiang original HiGHS LP."
                : fatal_msg
        );
    }

    return {bestHeight, bestU, bestParams};
}

static std::tuple<double, Eigen::VectorXd, JiangOriginalParams>
h_m_jiang_original_highs_full(const Eigen::MatrixXd& G, int m)
{
    return h_m_jiang_original_highs_full_impl(G, m);
}

double h_m_jiang_original_highs(const Eigen::MatrixXd& G, int m)
{
    return std::get<0>(h_m_jiang_original_highs_full(G, m));
}

#endif // HAVE_HIGHS
