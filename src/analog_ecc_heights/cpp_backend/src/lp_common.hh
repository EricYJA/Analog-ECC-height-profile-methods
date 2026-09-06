#pragma once

namespace m_height_lp {
constexpr double infinity = 1e20;
enum class Backend { Glpk, Highs };
enum class Status { Optimal, Infeasible, Unbounded };
struct Result { Status status; double value; };

} // namespace m_height_lp
