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
- `h_m_jiang_simplified_lp_highs_early_quit(G, m, early_quit_threshold, num_threads=16)`
- `h_m_jiang_original_lp_glpk(G, m)`
- `h_m_jiang_original_lp_highs(G, m, num_threads=16)`

The simplified methods cap the result at the supplied threshold. Pass
`float("inf")` to compute the full height. Both original methods use the same
enumeration and constraints and assume no zero column in `G`.
HiGHS methods are exposed only when HiGHS support is built.

These names replace the previous Jiang API. The ordinary simplified methods
and additional-constraint variant have been removed.

## Roth LP API

- `h_m_roth_primal_lp_glpk(G, m, early_quit_threshold)`
- `h_m_roth_dual_lp_glpk(G, m, early_quit_threshold)`
- `h_m_roth_primal_lp_highs(G, m, early_quit_threshold, num_threads=16)`
- `h_m_roth_dual_lp_highs(G, m, early_quit_threshold, num_threads=16)`

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

## HiGHS execution

All four HiGHS APIs accept `num_threads=16`, controlling OpenMP workers.
Each worker owns one reusable solver and direct row-wise `HighsLp` buffers;
each individual solve uses single-threaded simplex. `num_threads=1` executes
a serial loop without an OpenMP region. Positive thread counts are required;
builds without OpenMP fall back to serial execution.

Work is generated in bounded batches (at most four cases per requested worker),
with dynamic scheduling. Roth groups objectives sharing a complement matrix.
Threshold checks occur after completed solves; already-running LPs may finish.
Original Jiang has no threshold parameter. All HiGHS paths check solver options
and statuses and never fall back to GLPK. Presolve and simplex scaling are off.

C-contiguous float64 NumPy inputs are viewed without an input-matrix copy.
Other dtypes/layouts are converted by the Python binding as needed. LP coefficients
are constructed directly in the HiGHS arrays; no intermediate Eigen LP matrix
is used. HiGHS may still copy the model internally when loading it.

Sources are `methods_jiang_lp.cc` and `methods_roth_lp.cc`.

## Combinatorial API

Each method has one implementation for sequential and parallel execution.
All accept a final `num_threads=16` argument, after `tol=1e-10`:

- `h_m_roth_primal_combinatorial(G, m=None, tol=1e-10, num_threads=16)`
- `h_m_roth_primal_combinatorial_pruning(G, m, tol=1e-10, num_threads=16)`
- `h_m_roth_dual_combinatorial_generator(G, m, tol=1e-10, num_threads=16)`
- `h_m_roth_dual_combinatorial_parity(H, m, tol=1e-10, num_threads=16)`
- `h_m_roth_mds_combinatorial(G, m, tol=1e-10, num_threads=16)`
- `h_m_roth_mds_combinatorial_parity(H, m, tol=1e-10, num_threads=16)`

Set `num_threads=1` for sequential execution of the same algorithm, including
pruning. Nonpositive counts are rejected. Without OpenMP, the same code runs
sequentially regardless of the requested positive count. Candidate evaluation
uses dynamically scheduled OpenMP loops; distance/MDS prechecks remain serial.

The combinatorial APIs use suffix-free names; the former `_omp` exports have
been removed. Use `num_threads=1` for sequential execution. Existing `tol`
positional arguments remain valid. For primal combinatorial, integer `m` returns a float;
omitting `m` or passing `m=None` returns a list of heights for `m=1,...,n-k`.
The separate `_omp_all` API has been removed. Full-profile computation still
shares work across heights rather than calling the scalar method repeatedly.

```python
height = h_m_roth_primal_combinatorial(G, m=2, num_threads=1)
profile = h_m_roth_primal_combinatorial(G, m=None, num_threads=16)
```

C++ uses same-name overloads: pass integer `m` for a `double`, or
`std::nullopt` for `std::vector<double>`, followed by optional `tol` and
`num_threads` arguments.

The MDS generator method requires `m=n-k`; the MDS parity method requires
`m=H.shape[0]`. Both reject non-MDS input. For MDS parity, the dual-parity
formula specializes to the maximum row absolute sum of
`inverse(H[:, S]) @ H[:, complement(S)]` over all subsets of size `m`.

All combinatorial methods use full-pivot LU with the same parent-scale rank
threshold and linear solves rather than explicit inverses. Subset enumeration,
complements, column gathering, and factorization helpers are shared in
`utils.hh`. Public input validation is performed once per call.

Primal scalar, profile, and pruning paths share a transform built once per
candidate basis. Dual methods factor once per candidate basis within each outer
subset, solve multiple right-hand sides together, and retain a separate minimum
for each target before taking the maximum. MDS G/H share a short implementation,
using column/row absolute sums respectively.

Outer subset lists are still materialized, but there are no string-key caches,
global factorization caches, or additional workspace classes. Temporary
coefficient buffers are reused within target/sign loops. The candidate sets,
max/min nesting, strict primal admissibility comparisons, and pruning bounds
are unchanged; arithmetic reordering may produce small floating-point differences.

## References

1. Ron M. Roth, “[Analog Error-Correcting Codes](https://doi.org/10.1109/TIT.2020.2977918),” *IEEE Transactions on Information Theory*, 66(7), 4075–4088, 2020.
2. Anxiao Jiang, “[Analog Error-Correcting Codes: Designs and Analysis](https://doi.org/10.1109/TIT.2024.3454059),” *IEEE Transactions on Information Theory*, 70(11), 7740–7756, 2024.
3. Ron M. Roth, Ziyuan Zhu, Changcheng Yuan, Paul H. Siegel, and Anxiao Jiang, “[On the Height Profile of Analog Error-Correcting Codes](https://arxiv.org/abs/2602.20366),” *2026 IEEE International Symposium on Information Theory (ISIT)*, also available as arXiv:2602.20366.
