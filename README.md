# Analog ECC Height-Profile Methods

This repository contains Python and C++ methods for computing the $m$-height of analog error-correcting codes used in analog in-memory computing. The implementations include linear-programming and combinatorial approaches.

All implementations are based on the methods and theorems developed in the three papers listed below.

## Repository layout

- `m-height_calculation/python_module/`: optimized C++ implementations exposed to Python with pybind11.

## Build the Python module

The C++ module requires Python 3.10 or later, CMake, a C++17 compiler, Eigen, GLPK, pybind11, and NumPy. HiGHS, OpenMP, and CUDA support are optional.

```bash
cd m-height_calculation/python_module
python -m pip install .
```

Example:

```python
import numpy as np
from solve_m_height_cpp import h_m_jiang_simplified_lp_glpk_early_quit

G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
h_m = h_m_jiang_simplified_lp_glpk_early_quit(G, 1, float("inf"))
print(h_m)
```

## Jiang LP API

- `h_m_jiang_simplified_lp_glpk_early_quit(G, m, early_quit_threshold)`
- `h_m_jiang_simplified_lp_highs_early_quit(G, m, early_quit_threshold)`
- `h_m_jiang_original_lp_glpk(G, m)`
- `h_m_jiang_original_lp_highs(G, m)`

The simplified methods cap the result at the supplied threshold. Pass
`float("inf")` to compute the full height. Both original methods use the same
enumeration and constraints and assume no zero column in `G`.
HiGHS methods are exposed only when HiGHS support is built.

These names replace the previous Jiang API. The ordinary simplified methods
and additional-constraint variant have been removed.

## Roth LP API

- `h_m_roth_primal_lp_glpk(G, m, early_quit_threshold)`
- `h_m_roth_dual_lp_glpk(G, m, early_quit_threshold)`
- `h_m_roth_primal_lp_highs(G, m, early_quit_threshold)`
- `h_m_roth_dual_lp_highs(G, m, early_quit_threshold)`

All four return `min(h_m, early_quit_threshold)`; pass `float("inf")` for
the full height. Early exit happens after a completed LP solve. For `m=0`,
the result is `min(1, early_quit_threshold)`. NaN thresholds are rejected.
HiGHS methods are available only in builds with HiGHS support. Each method
uses its named backend without fallback. These replace the old Roth LP APIs,
including the dominance-constraint variant.

Jiang original and Roth primal/dual follow the simplified Jiang LP object
lifecycle: GLPK creates and deletes each problem; HiGHS reuses its solver and
allocated model buffers within a call, clearing and reloading each LP without
preserving its simplex basis. All GLPK calls share one mutex.

All GLPK methods enumerate cases sequentially and incrementally, use the same
Eigen-to-GLPK row-loading path, and reuse row scratch arrays within each LP.
They check simplex return codes and require optimal status for finite results.
The simplified GLPK method tracks only the scalar height and stops generating
cases when its threshold is exceeded. It requires finite input,
`1 <= m <= min(30, n-1)`, and a non-NaN threshold.

## References

1. Ron M. Roth, “[Analog Error-Correcting Codes](https://doi.org/10.1109/TIT.2020.2977918),” *IEEE Transactions on Information Theory*, 66(7), 4075–4088, 2020.
2. Anxiao Jiang, “[Analog Error-Correcting Codes: Designs and Analysis](https://doi.org/10.1109/TIT.2024.3454059),” *IEEE Transactions on Information Theory*, 70(11), 7740–7756, 2024.
3. Ron M. Roth, Ziyuan Zhu, Changcheng Yuan, Paul H. Siegel, and Anxiao Jiang, “[On the Height Profile of Analog Error-Correcting Codes](https://arxiv.org/abs/2602.20366),” *2026 IEEE International Symposium on Information Theory (ISIT)*, also available as arXiv:2602.20366.
