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
| `h_m_jiang_original_lp` | `1 <= m <= min(30, n-1)` | No zero column in `G`; no threshold option |
| `h_m_jiang_simplified_lp` | `1 <= m <= min(30, n-1)` | Supports threshold capping |
| `h_m_roth_primal_lp` | `0 <= m < n` | At `m=0`, returns `min(1, threshold)` |
| `h_m_roth_dual_lp` | `0 <= m < n` | At `m=0`, returns `min(1, threshold)` |

LP matrices must have at least one row and one column. Unlike the combinatorial APIs, LP calls do not require full row rank. The 30-coordinate limit applies to the **height index `m` in Jiang's methods**, not the number of columns in `G`.

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

### Full profiles

Only `h_m_roth_primal_combinatorial` accepts `m=None`, which is also its default. It returns a `list[float]`:

```text
[h_1, h_2, ..., h_(n-k)]
```

The list does not include `h_0`; its length is the redundancy `n-k`. A square full-rank generator has an empty profile. Full profiles share computation across indices. Integer `m` returns a scalar `float`.

```python
from analog_ecc_heights import h_m_roth_primal_combinatorial

G = [[1.0, -2.0, 4.0]]
print(h_m_roth_primal_combinatorial(G))     # [2.0, 4.0]
print(h_m_roth_primal_combinatorial(G, 2))  # 4.0
```

### MDS specializations

`h_m_roth_mds_combinatorial` requires `m=n-k`; its parity counterpart requires `m=r`. Both check the MDS condition under `tol` and reject non-MDS input. **At zero redundancy, these two specializations return `0.0`**. This is distinct from the general methods' `m=0` convention of `1.0`.

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

LP feasibility and optimality tolerances are determined by the selected solver; LP functions do not accept `tol`. Nearly singular matrices, extreme coefficient scales, and different solver versions can produce different results or classifications across methods and backends. Compare finite results with a suitable numerical tolerance and handle positive infinity separately.

## Errors and compatibility

Wrong option types raise `TypeError`; invalid matrix values, index ranges, backend combinations, rank conditions, and unsupported thread requests raise `ValueError`. Missing native support raises `ImportError` with installation guidance. LP solver failures raise `RuntimeError`. `m` and `num_threads` must be integer values; booleans are rejected.

`solve_m_height_cpp` remains available for existing code after installing the corresponding native backends. Its original function names, positional options, and native defaults are supported. Import from `analog_ecc_heights` to use the backend selection and option validation described in this guide.
