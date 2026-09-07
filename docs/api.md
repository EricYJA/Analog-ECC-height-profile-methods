# Python API

Import public functions from `analog_ecc_heights`. The matrix and height index can be positional; all configuration options are keyword-only.

A generator matrix `G` has shape `(k, n)` and generates codewords `u @ G`. A parity-check matrix `H` has shape `(r, n)` and describes codewords satisfying `H @ c = 0`. Matrix inputs must be real, finite, two-dimensional arrays or convertible array-like values. They are converted to float64 as needed; caller data is not modified. Contiguous, strided, and read-only arrays are supported.

## Linear-programming methods

```python
h_m_jiang_original_lp(G, m, *, backend="python", num_threads=None)

h_m_jiang_simplified_lp(
    G, m, *, backend="python", early_quit_threshold=float("inf"), num_threads=None
)

h_m_roth_primal_lp(
    G, m, *, backend="python", early_quit_threshold=float("inf"), num_threads=None
)

h_m_roth_dual_lp(
    G, m, *, backend="python", early_quit_threshold=float("inf"), num_threads=None
)
```

All four accept `backend="python"`, `"cpp-glpk"`, or `"cpp-highs"` and return a Python `float`.

| Function | Index restriction | Additional behavior |
| --- | --- | --- |
| `h_m_jiang_original_lp` | `1 <= m < n` | No zero column in `G`; no threshold option |
| `h_m_jiang_simplified_lp` | `1 <= m < n` | Supports threshold capping |
| `h_m_roth_primal_lp` | `0 <= m < n` | At `m=0`, returns `min(h_0, threshold)` |
| `h_m_roth_dual_lp` | `0 <= m < n` | At `m=0`, returns `min(h_0, threshold)` |

For the Roth LP methods, `h_0=0` when `G` is exactly all zero, and `h_0=1` otherwise, including nonzero matrices without full row rank. This follows the zero-vector convention in Roth (2020); the threshold cap still applies.

LP matrices must have at least one row and one column. Unlike the combinatorial APIs, LP calls do not require full row rank. Jiang's methods accept every integer index `1 <= m < n`; there is no fixed cap of 30. Original Jiang enumerates `2**m` sign patterns per ordering, so large indices can still require substantial computation.

### Thresholds and solver outcomes

The three threshold-capable methods return `min(h_m, early_quit_threshold)`. The default positive infinity computes the full height. A completed inner solve that exceeds the threshold permits early exit; a result equal to the threshold can still require additional enumeration. The reported value is capped, rather than an uncapped lower bound.

Real thresholds, including negative values and infinities, are accepted; NaN is rejected. Original Jiang has no threshold argument. Positive infinity represents an unbounded height. Infeasible sign/order cases are ignored in original Jiang; infeasibility of a Roth dual inner problem certifies an infinite height. Unexpected solver failures raise an error and do not produce a height or trigger a backend switch.

```python
from analog_ecc_heights import h_m_roth_primal_lp

G = [[1.0, -2.0, 4.0]]
assert abs(h_m_roth_primal_lp(G, 1) - 2.0) < 1e-8
assert h_m_roth_primal_lp(G, 1, early_quit_threshold=1.5) == 1.5
```

## Combinatorial methods

```python
h_m_roth_primal_combinatorial(
    G, m=None, *, backend="python", tol=1e-10, num_threads=None
)

h_m_roth_primal_combinatorial_pruning(
    G, m, *, backend="python", tol=1e-10, num_threads=None
)

h_m_roth_dual_combinatorial_generator(
    G, m, *, backend="python", tol=1e-10, num_threads=None
)

h_m_roth_dual_combinatorial_parity(
    H, m, *, backend="python", tol=1e-10, num_threads=None
)

h_m_roth_mds_combinatorial(
    G, m, *, backend="python", tol=1e-10, num_threads=None
)

h_m_roth_mds_combinatorial_parity(
    H, m, *, backend="python", tol=1e-10, num_threads=None
)
```

All six accept `backend="python"` or `"cpp"`. No LP solver name is selected for combinatorial calls.

Generator matrices must have `1 <= k <= n` and full row rank under `tol`. Parity-check matrices must have `0 <= r < n`, at least one column, and full row rank under `tol`; a `(0, n)` parity-check matrix is allowed. `tol` must be finite and nonnegative. It controls complete-pivot LU rank decisions using the parent matrix's scale; see [numerical behavior](#numerical-behavior).

Scalar indices satisfy `0 <= m < n`. General combinatorial methods return `1.0` at `m=0`, a finite height when supported by the code's minimum distance, or positive infinity. Pruning computes the same scalar quantity as the primal method while eliminating candidates using bounds.

### Finite profiles

Only `h_m_roth_primal_combinatorial` accepts `m=None`, which is also its default. It returns a `list[float]`:

```text
[h_1, h_2, ..., h_(d-1)]
```

Here `d` is the minimum distance determined under `tol`. The list has length `d-1` and contains only the finite positive-index heights; `h_0` and the infinite heights starting at `h_d` are omitted. For an MDS code, `d-1 = n-k`. Any code with `d=1`, including a square full-rank generator, has an empty profile. Profiles share computation across indices. Integer `m` returns a scalar `float`, including positive infinity when `m >= d`.

This changes the earlier `m=None` result of length `n-k`: non-MDS profiles no longer include trailing infinities.

```python
from analog_ecc_heights import h_m_roth_primal_combinatorial

G = [[1.0, -2.0, 4.0]]
print(h_m_roth_primal_combinatorial(G))     # [2.0, 4.0]
print(h_m_roth_primal_combinatorial(G, 2))  # 4.0

G = [[0.0, 1.0, 2.0]]  # d=2, redundancy=2
print(h_m_roth_primal_combinatorial(G))     # [2.0]
print(h_m_roth_primal_combinatorial(G, 2))  # inf
```

### MDS specializations

`h_m_roth_mds_combinatorial` requires `m=n-k`; its parity counterpart requires `m=r`. Both check the MDS condition under `tol` and reject non-MDS input. At zero redundancy, both return `1.0`: the code is the full space and has `h_0=1`, consistent with the general methods.

## Backend availability and threads

Every method defaults to `backend="python"`, even when native support is installed. Choose a backend explicitly to use a different implementation:

| Backend | Supported methods | Implementation |
| --- | --- | --- |
| `python` | LP and combinatorial | NumPy/SciPy; LP uses SciPy's HiGHS solver |
| `cpp` | Combinatorial | C++ |
| `cpp-glpk` | LP | C++ with GLPK |
| `cpp-highs` | LP | C++ with HiGHS |

Install optional backends using the [installation guide](installation.md#install-optional-native-backends). Check which backends are available in your environment:

```python
from analog_ecc_heights import available_backends

print(available_backends())  # ["python"] in a default installation
```

`available_backends()` returns a `list[str]` containing `"python"` and whichever native backends can be loaded. An unavailable backend or solver failure raises an error; calls never switch to another backend automatically.

Pass `num_threads=1` for sequential execution. Leaving it as `None` uses the backend default:

| Backend | Accepted `num_threads` | Meaning of `None` |
| --- | --- | --- |
| `python` | `None` or `1` | Sequential Python enumeration |
| `cpp-glpk` | `None` or `1` | Sequential GLPK enumeration |
| `cpp`, `cpp-highs` | Positive integers; OpenMP required at build time | 16 workers |

The parameter controls this package's workers. It does not set every NumPy/SciPy or BLAS internal thread count. Unsupported thread counts raise an error; an explicit parallel request is not silently ignored.

Large problems can require exponentially many cases. Compare performance using matrices and height indices representative of your application.

## Numerical behavior

`tol` applies only to combinatorial methods. Rank decisions use complete-pivot LU with cutoff `tol * reference_scale`, where the reference scale is the largest absolute entry of the input matrix. Submatrices use their parent matrix's scale. Changing `tol` can change rank and MDS classifications, and whether a reported height is finite.

The C++ HiGHS wrapper accepts recoverable API warnings, such as dropping coefficients below the solver's matrix tolerance, then checks the final model status. Errors, incomplete solves, ambiguous statuses, and nonfinite optimal objectives raise `RuntimeError`.

LP feasibility and optimality tolerances are determined by the selected solver; LP functions do not accept `tol`. Nearly singular matrices, extreme coefficient scales, and different solver versions can produce different results or classifications across methods and backends. Compare finite results with a suitable numerical tolerance and handle positive infinity separately.

## Errors and compatibility

Wrong option types raise `TypeError`; invalid matrix values, index ranges, backend combinations, rank conditions, and unsupported thread requests raise `ValueError`. Missing native support raises `ImportError` with installation guidance. LP solver failures raise `RuntimeError`. `m` and `num_threads` must be integer values; booleans are rejected.

`solve_m_height_cpp` remains available for existing code after installing the corresponding native backends. Its original function names, positional options, and native defaults are supported. Import from `analog_ecc_heights` to use the backend selection and option validation described in this guide.

## Relationship to the papers

Roth (2020) supplies the height definition, including the zero-vector convention. For nontrivial codes, the finite positive-index heights have `1 <= m < d`; `h_d` is infinite. The `m=None` API returns this finite prefix. The paper's complete height profile also includes `h_0` and the infinite entries.

The Roth methods map to [Roth et al. (2026), arXiv v1](https://arxiv.org/html/2602.20366v1) as follows. This mapping applies to both Python and C++.

| Function | Formula |
| --- | --- |
| `h_m_roth_primal_lp` | Equation (4) |
| `h_m_roth_dual_lp` | Equations (5)–(7) |
| `h_m_roth_primal_combinatorial` | Theorem 5, equation (15) |
| `h_m_roth_primal_combinatorial_pruning` | Equation (15), with sign symmetry and an upper bound for pruning |
| `h_m_roth_dual_combinatorial_generator` | Theorem 8, equation (20) |
| `h_m_roth_dual_combinatorial_parity` | Theorem 8, equation (21) |
| `h_m_roth_mds_combinatorial` and its parity counterpart | Corollary 11 |

The original Jiang method implements the sign/order LP enumeration. Its constraints also appear in Theorem 2 of the author's [NVMW 2024 extended abstract](https://nvmw.ucsd.edu/nvmw2024-program/nvmw2024-paper23-final_version_your_extended_abstract.pdf).

`h_m_jiang_simplified_lp` is a derived variant: each LP adds ordering inequalities to a Roth primal feasible region. Conversely, an extremal codeword can be sign-reversed and assigned a distinguished coordinate of largest magnitude that satisfies those inequalities, so the overall maximum is unchanged. The function name is retained for compatibility; these exact simplified constraints have not been verified against the final Jiang journal text.

See the [references](../README.md#references), including the correction to Roth (2020), for the source papers.
