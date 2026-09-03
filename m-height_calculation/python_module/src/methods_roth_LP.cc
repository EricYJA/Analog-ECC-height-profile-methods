#include "methods.hh"

#ifdef HAVE_HIGHS
#include <highs/Highs.h>
#endif

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kHighsInf = 1e20;

void validate_nonempty_matrix(const Eigen::MatrixXd& M, const char* name) {
    if (M.rows() == 0 || M.cols() == 0) {
        throw std::runtime_error(std::string(name) + " must be a non-empty 2D matrix.");
    }
}

void validate_m_for_columns(int m, int n) {
    if (m < 0 || m > n - 1) {
        std::ostringstream oss;
        oss << "m must satisfy 0 <= m <= n-1, got m=" << m << ", n=" << n;
        throw std::invalid_argument(oss.str());
    }
}

template <typename Fn>
bool for_each_subset_until(int n, int subset_size, Fn&& fn) {
    if (subset_size < 0 || subset_size > n) return true;

    std::vector<int> subset;
    subset.reserve(subset_size);

    std::function<bool(int, int)> dfs = [&](int start, int need) {
        if (need == 0) {
            return fn(subset);
        }
        for (int v = start; v <= n - need; ++v) {
            subset.push_back(v);
            const bool keep_going = dfs(v + 1, need - 1);
            subset.pop_back();
            if (!keep_going) return false;
        }
        return true;
    };
    return dfs(0, subset_size);
}

std::vector<int> complement_indices(int n, const std::vector<int>& subset) {
    std::vector<char> in_subset(n, 0);
    for (int idx : subset) {
        if (idx < 0 || idx >= n) throw std::out_of_range("subset index out of range");
        in_subset[idx] = 1;
    }
    std::vector<int> out;
    out.reserve(n - static_cast<int>(subset.size()));
    for (int i = 0; i < n; ++i) {
        if (!in_subset[i]) out.push_back(i);
    }
    return out;
}

#ifdef HAVE_HIGHS
void configure_highs(Highs& highs) {
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("log_to_console", false);
    highs.setOptionValue("threads", 1);
    highs.setOptionValue("solver", "simplex");
    highs.setOptionValue("simplex_strategy", 0);
    highs.setOptionValue("presolve", "off");
    highs.setOptionValue("scaling", "off");
    highs.setOptionValue("infinite_bound", kHighsInf);
    highs.setOptionValue("infinite_cost", kHighsInf);
}

enum class RothLpKind {
    kPrimal,
    kDual,
};

double run_highs_lp(Highs& highs, HighsLp& lp, RothLpKind kind) {
    if (highs.clearModel() != HighsStatus::kOk) {
        throw std::runtime_error("HiGHS failed to clear the previous Roth LP model.");
    }
    if (highs.passModel(lp) != HighsStatus::kOk) {
        throw std::runtime_error("HiGHS failed to load a Roth LP model.");
    }
    if (highs.run() != HighsStatus::kOk) {
        throw std::runtime_error("HiGHS failed while solving a Roth LP model.");
    }

    const HighsModelStatus status = highs.getModelStatus();
    if (status == HighsModelStatus::kOptimal) {
        return highs.getInfo().objective_function_value;
    }
    if (status == HighsModelStatus::kUnboundedOrInfeasible) {
        // The primal models are always feasible (u = 0), while the dual
        // objective is bounded below. In either case this ambiguous status
        // represents an infinite primal value.
        return std::numeric_limits<double>::infinity();
    }
    if (kind == RothLpKind::kPrimal && status == HighsModelStatus::kUnbounded) {
        return std::numeric_limits<double>::infinity();
    }
    if (kind == RothLpKind::kDual && status == HighsModelStatus::kInfeasible) {
        // Infeasibility of the l1 representation LP is dual to an unbounded
        // primal maximization problem.
        return std::numeric_limits<double>::infinity();
    }

    std::ostringstream oss;
    oss << "HiGHS returned unexpected Roth "
        << (kind == RothLpKind::kPrimal ? "primal" : "dual")
        << " LP status: " << highs.modelStatusToString(status);
    throw std::runtime_error(oss.str());
}

class PrimalLpWorkspace {
public:
    PrimalLpWorkspace(int k, int n, int m, bool add_dominance_constraints)
        : k_(k),
          complement_size_(n - m),
          bound_rows_(2 * complement_size_),
          rows_(bound_rows_ + (add_dominance_constraints ? m - 1 : 0)),
          add_dominance_constraints_(add_dominance_constraints)
    {
        configure_highs(highs_);

        lp_.num_col_ = k_;
        lp_.num_row_ = rows_;
        lp_.col_cost_.assign(k_, 0.0);
        lp_.col_lower_.assign(k_, -kHighsInf);
        lp_.col_upper_.assign(k_, kHighsInf);
        lp_.row_lower_.assign(rows_, -kHighsInf);
        lp_.row_upper_.assign(rows_, 1.0);
        for (int row = bound_rows_; row < rows_; ++row) {
            lp_.row_upper_[row] = 0.0;
        }

        lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
        lp_.a_matrix_.start_.resize(rows_ + 1);
        lp_.a_matrix_.index_.resize(
            static_cast<size_t>(rows_) * static_cast<size_t>(k_));
        lp_.a_matrix_.value_.assign(
            static_cast<size_t>(rows_) * static_cast<size_t>(k_), 0.0);
        for (int row = 0; row <= rows_; ++row) {
            lp_.a_matrix_.start_[row] = static_cast<HighsInt>(row * k_);
        }
        for (int row = 0; row < rows_; ++row) {
            const int base = row * k_;
            for (int col = 0; col < k_; ++col) {
                lp_.a_matrix_.index_[base + col] = col;
            }
        }
        lp_.sense_ = ObjSense::kMaximize;
    }

    void prepare_subset(const Eigen::MatrixXd& G, const std::vector<int>& complement) {
        if (static_cast<int>(complement.size()) != complement_size_) {
            throw std::logic_error("Roth primal LP complement size mismatch.");
        }

        for (int t = 0; t < complement_size_; ++t) {
            const int j = complement[t];
            const int base_pos = t * k_;
            const int base_neg = (complement_size_ + t) * k_;
            for (int col = 0; col < k_; ++col) {
                lp_.a_matrix_.value_[base_pos + col] = G(col, j);
                lp_.a_matrix_.value_[base_neg + col] = -G(col, j);
            }
        }
    }

    double solve_for_i(const Eigen::MatrixXd& G, const std::vector<int>& S, int i) {
        for (int col = 0; col < k_; ++col) {
            lp_.col_cost_[col] = G(col, i);
        }

        if (add_dominance_constraints_) {
            int row = bound_rows_;
            for (int j : S) {
                if (j == i) continue;
                const int base = row * k_;
                for (int col = 0; col < k_; ++col) {
                    lp_.a_matrix_.value_[base + col] = G(col, j) - G(col, i);
                }
                ++row;
            }
            if (row != rows_) {
                throw std::logic_error("Roth primal LP dominance row mismatch.");
            }
        }

        return run_highs_lp(highs_, lp_, RothLpKind::kPrimal);
    }

private:
    int k_;
    int complement_size_;
    int bound_rows_;
    int rows_;
    bool add_dominance_constraints_;
    Highs highs_;
    HighsLp lp_;
};

class DualLpWorkspace {
public:
    DualLpWorkspace(int k, int n, int m)
        : k_(k), complement_size_(n - m), vars_(2 * complement_size_)
    {
        configure_highs(highs_);

        lp_.num_col_ = vars_;
        lp_.num_row_ = k_;
        lp_.col_cost_.assign(vars_, 1.0);
        lp_.col_lower_.assign(vars_, 0.0);
        lp_.col_upper_.assign(vars_, kHighsInf);
        lp_.row_lower_.assign(k_, 0.0);
        lp_.row_upper_.assign(k_, 0.0);

        lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
        lp_.a_matrix_.start_.resize(k_ + 1);
        lp_.a_matrix_.index_.resize(
            static_cast<size_t>(k_) * static_cast<size_t>(vars_));
        lp_.a_matrix_.value_.assign(
            static_cast<size_t>(k_) * static_cast<size_t>(vars_), 0.0);
        for (int row = 0; row <= k_; ++row) {
            lp_.a_matrix_.start_[row] = static_cast<HighsInt>(row * vars_);
        }
        for (int row = 0; row < k_; ++row) {
            const int base = row * vars_;
            for (int col = 0; col < vars_; ++col) {
                lp_.a_matrix_.index_[base + col] = col;
            }
        }
        lp_.sense_ = ObjSense::kMinimize;
    }

    void prepare_subset(const Eigen::MatrixXd& G, const std::vector<int>& complement) {
        if (static_cast<int>(complement.size()) != complement_size_) {
            throw std::logic_error("Roth dual LP complement size mismatch.");
        }

        for (int row = 0; row < k_; ++row) {
            const int base = row * vars_;
            for (int t = 0; t < complement_size_; ++t) {
                const int j = complement[t];
                lp_.a_matrix_.value_[base + t] = G(row, j);
                lp_.a_matrix_.value_[base + complement_size_ + t] = -G(row, j);
            }
        }
    }

    double solve_for_i(const Eigen::MatrixXd& G, int i) {
        for (int row = 0; row < k_; ++row) {
            lp_.row_lower_[row] = G(row, i);
            lp_.row_upper_[row] = G(row, i);
        }
        return run_highs_lp(highs_, lp_, RothLpKind::kDual);
    }

private:
    int k_;
    int complement_size_;
    int vars_;
    Highs highs_;
    HighsLp lp_;
};

double roth_primal_lp_value(
    const Eigen::MatrixXd& G,
    int m,
    bool add_dominance_constraints)
{
    // The project-wide convention is h_0(C) = 1.
    if (m == 0) return 1.0;

    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    PrimalLpWorkspace workspace(k, n, m, add_dominance_constraints);

    double best = -std::numeric_limits<double>::infinity();
    for_each_subset_until(n, m, [&](const std::vector<int>& S) {
        const std::vector<int> complement = complement_indices(n, S);
        workspace.prepare_subset(G, complement);

        for (int i : S) {
            best = std::max(best, workspace.solve_for_i(G, S, i));
            if (best == std::numeric_limits<double>::infinity()) {
                return false;
            }
        }
        return true;
    });
    return best;
}

double roth_dual_lp_value(const Eigen::MatrixXd& G, int m) {
    // The project-wide convention is h_0(C) = 1.
    if (m == 0) return 1.0;

    const int k = static_cast<int>(G.rows());
    const int n = static_cast<int>(G.cols());
    DualLpWorkspace workspace(k, n, m);

    double best = -std::numeric_limits<double>::infinity();
    for_each_subset_until(n, m, [&](const std::vector<int>& S) {
        const std::vector<int> complement = complement_indices(n, S);
        workspace.prepare_subset(G, complement);

        for (int i : S) {
            best = std::max(best, workspace.solve_for_i(G, i));
            if (best == std::numeric_limits<double>::infinity()) {
                return false;
            }
        }
        return true;
    });
    return best;
}
#endif

}  // namespace

double h_m_roth_primal_lp(const Eigen::MatrixXd& G, int m) {
    validate_nonempty_matrix(G, "G");
    validate_m_for_columns(m, static_cast<int>(G.cols()));

#ifndef HAVE_HIGHS
    (void)G;
    (void)m;
    throw std::runtime_error("h_m_roth_primal_lp requires HiGHS support (build with HAVE_HIGHS).");
#else
    return roth_primal_lp_value(G, m, false);
#endif
}

double h_m_roth_primal_lp_constraint(const Eigen::MatrixXd& G, int m) {
    validate_nonempty_matrix(G, "G");
    validate_m_for_columns(m, static_cast<int>(G.cols()));

#ifndef HAVE_HIGHS
    (void)G;
    (void)m;
    throw std::runtime_error("h_m_roth_primal_lp_constraint requires HiGHS support (build with HAVE_HIGHS).");
#else
    return roth_primal_lp_value(G, m, true);
#endif
}

double h_m_roth_dual_lp(const Eigen::MatrixXd& G, int m) {
    validate_nonempty_matrix(G, "G");
    validate_m_for_columns(m, static_cast<int>(G.cols()));

#ifndef HAVE_HIGHS
    (void)G;
    (void)m;
    throw std::runtime_error("h_m_roth_dual_lp requires HiGHS support (build with HAVE_HIGHS).");
#else
    return roth_dual_lp_value(G, m);
#endif
}
