#include "methods.hh"
#include "lp_workspace.hh"

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

// Enumerate (a, X) lazily: X contains m-1 coordinates other than a,
// and Y is the complement of {a} union X. GLPK solves remain sequential.
static double jiang_simplified_glpk_value(
    const Eigen::MatrixXd& G, int m, double early_quit_threshold)
{
    const int k = G.rows(), n = G.cols();
    if (k == 0 || n == 0 || !G.allFinite())
        throw std::invalid_argument("G must be a finite, nonempty matrix.");
    if (m < 1 || m >= n || m > 30)
        throw std::invalid_argument("Simplified Jiang GLPK requires 1 <= m <= min(30, n-1).");
    if (std::isnan(early_quit_threshold))
        throw std::invalid_argument("early_quit_threshold must not be NaN.");

    m_height_lp::Workspace lp(m - 1 + 2 * (n - m), k,
                             m_height_lp::Backend::Glpk, true);
    lp.upper.setOnes();
    lp.upper.head(m - 1).setZero();
    double best = -std::numeric_limits<double>::infinity();
    bool stop = false;

    for (int a = 0; a < n && !stop; ++a) {
        lp.cost = G.col(a);
        std::vector<int> remaining, X;
        for (int j = 0; j < n; ++j)
            if (j != a) remaining.push_back(j);
        std::function<void(int, int)> enumerate = [&](int start, int need) {
            if (stop) return;
            if (need != 0) {
                for (int t = start; t <= static_cast<int>(remaining.size()) - need && !stop; ++t) {
                    X.push_back(remaining[t]);
                    enumerate(t + 1, need - 1);
                    X.pop_back();
                }
                return;
            }

            int row = 0;
            // c_x <= c_a for x in X.
            for (int x : X)
                lp.A.row(row++) = (G.col(x) - G.col(a)).transpose();
            // -1 <= c_y <= 1 for y in Y.
            for (int y : remaining) {
                if (std::binary_search(X.begin(), X.end(), y)) continue;
                lp.A.row(row++) = G.col(y).transpose();
                lp.A.row(row++) = -G.col(y).transpose();
            }

            const auto result = lp.solve();
            if (result.status == m_height_lp::Status::Optimal)
                best = std::max(best, result.value);
            else if (result.status == m_height_lp::Status::Unbounded)
                best = std::numeric_limits<double>::infinity();
            else
                throw std::runtime_error("Simplified Jiang LP unexpectedly infeasible (u=0 is feasible).");
            stop = best > early_quit_threshold || std::isinf(best);
        };
        enumerate(0, m - 1);
    }
    return std::min(best, early_quit_threshold);
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
            const double value = jiang_simplified_glpk_value(G, m, early_quit_threshold);
            // The public API returns only the height; GLPK no longer collects
            // a maximizing vector or subset metadata.
            return {value, Eigen::VectorXd::Zero(k), {}};
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
    return jiang_simplified_glpk_value(G, m, early_quit_threshold);
}

#ifdef HAVE_HIGHS
double h_m_jiang_simplified_lp_highs_early_quit(const Eigen::MatrixXd& G,
                                     int m,
                                     double early_quit_threshold) {
    return std::get<0>(h_m_jiang_simplified_lp_highs_early_quit_full(G, m, early_quit_threshold));
}
#endif


namespace {
using OriginalBackend = m_height_lp::Backend;

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
    m_height_lp::Workspace lp(rows, k, backend, true);
    auto& A = lp.A;
    auto& bounds = lp.upper;
    auto& cost = lp.cost;
    lp.lower[equality_row] = 1.0;
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
                    const auto result = lp.solve();
                    if (result.status == m_height_lp::Status::Optimal)
                        best = std::max(best, result.value);
                    else if (result.status == m_height_lp::Status::Unbounded)
                        best = std::numeric_limits<double>::infinity();
                    // Infeasible sign/order cases contribute nothing.
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
