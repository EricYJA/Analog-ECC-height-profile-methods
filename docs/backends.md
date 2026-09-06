# Backend behavior

The function name selects the mathematical formulation; `backend` selects its implementation. Every public function defaults to `"python"`, even when native support is installed. There is no automatic fallback when a requested backend is missing or a solver fails.

| Backend | Implementation | Native installation |
| --- | --- | --- |
| `python` | Python enumeration, NumPy combinatorial operations, SciPy LP | None |
| `cpp` | Existing C++/Eigen combinatorial methods | Eigen and GLPK |
| `cpp-glpk` | Existing C++ enumeration and GLPK LP solves | Eigen and GLPK |
| `cpp-highs` | Existing C++ enumeration and direct HiGHS LP solves | Eigen, GLPK, and HiGHS |

These backends belong to the same `analog-ecc-heights` distribution. The native options preserve the existing shared extension, including its cumulative GLPK dependency. Eigen is needed only to compile. HiGHS support is enabled explicitly at build time, and the system C++ library is required; the `highspy` Python package is not required.

`"cpp"` is only valid for combinatorial functions. `"cpp-glpk"` and `"cpp-highs"` are only valid for LP functions. See [installation](installation.md) for source-build flags.

## LP execution

The Python backend uses `scipy.optimize.linprog(method="highs")`. It translates each enumerated problem into SciPy arrays and explicitly selects free or nonnegative variables as required by the formulation. Enumeration is lazy; capped calls can stop without allocating all subsets. **Presolve is disabled** because it can misclassify a feasible, unbounded homogeneous LP as infeasible. A regression test covers this case. An exactly zero equality row with a nonzero right-hand side is certified infeasible directly; this avoids an unknown-status result in older HiGHS versions without introducing a numerical rank tolerance.

GLPK creates and deletes each problem, with calls protected by the shared native mutex. Enumeration is sequential. The HiGHS backend reuses each worker's solver and model buffers within a call, clearing and loading each LP without preserving a simplex basis. Native HiGHS solves use single-threaded simplex, with presolve and simplex scaling disabled; parallelism is across enumerated cases.

Original Jiang ignores infeasible sign/order cases. Roth dual maps an infeasible inner LP to an infinite height. The other feasible primal problems must return an optimum or an unbounded status. Iteration limits, numerical errors, rejected models, and unexpected statuses raise an error. Only completed inner optima certify finite values; an unfinished minimization cannot justify an early height cap.

Threshold-capable methods return `min(height, threshold)`. Parallel calls may finish some work that was already running when a threshold was reached. Original Jiang does not support a threshold.

## Combinatorial execution

The Python methods implement the same primal, pruning, dual generator/parity, and MDS algorithms as the native sources. Complete-pivot LU supplies rank decisions and basis solves. Full profiles reuse candidate computations across heights. Python subset enumeration is incremental; the preserved native combinatorial implementation may materialize outer subset lists.

Both implementations retain the matrix-specific restrictions and the native zero-redundancy MDS convention described in the [API guide](api.md). Large dimensions can still require an exponential number of candidates. No relative performance claim is implied by a backend name; measure the methods on the matrices and indices relevant to your application.

## Threading

`num_threads=None` selects the backend default:

- Python and GLPK use sequential enumeration and accept only `None` or `1`.
- Native combinatorial and HiGHS use 16 workers when built with OpenMP and 1 otherwise.
- Explicit counts greater than 1 for native methods require OpenMP; the public API rejects them if it is absent.

For repeatable serial comparisons, pass `num_threads=1` to every backend. The parameter governs this package's workers and does not promise control of NumPy/SciPy/BLAS internal threads. Native HiGHS work is buffered in batches bounded by the number of workers, rather than a complete task list.

## Numerical behavior

The public API converts real inputs to float64 as needed and does not modify them. Supported layouts include C/F-contiguous, strided, and read-only arrays; native bindings may copy inputs to satisfy their layout requirements.

`tol` belongs to the **combinatorial** API. Rank uses complete-pivot LU with cutoff `tol * reference_scale`, with a submatrix's reference scale taken from its parent matrix's largest absolute entry. This follows the existing Eigen implementation and avoids changing minimum-distance decisions by substituting NumPy's default SVD rank test. Basis computations use linear solves rather than explicit inverses.

LP feasibility and optimality tolerances belong to the selected solver; `tol` is not an LP argument. SciPy and system GLPK/HiGHS may also use different solver versions and internal scaling rules. Nearly singular matrices, extreme coefficient scales, or values near a rank/feasibility threshold can therefore produce different results or classifications across methods. Floating-point arithmetic order may change low-order digits even for ordinary inputs. Compare finite results with suitable numerical tolerances, and treat positive infinity separately.

The test suite checks known answers, generator/parity agreement, formulation agreement, threshold caps, solver errors, and rank-sensitive cases. These checks establish behavior for the covered cases; they do not make all solver versions numerically identical.

## Existing native callers

`import solve_m_height_cpp` is retained by a compatibility module in the same distribution. It loads the packaged native extension and exposes its original symbols. It requires native support and retains the original binding behavior, including legacy threading semantics. New code can use `analog_ecc_heights` for common signatures, explicit backend selection, and strict thread validation.
