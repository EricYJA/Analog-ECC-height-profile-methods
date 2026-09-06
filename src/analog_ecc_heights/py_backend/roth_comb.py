"""Roth's primal and dual combinatorial height formulas."""

import numpy as np

from ._linalg import (
    factor_columns,
    generator_minimum_distance_exceeds_validated,
    generator_minimum_distance_validated,
    matrix_scale,
    parity_check_minimum_distance_exceeds_validated,
    validate_generator_matrix,
    validate_m_for_columns,
    validate_parity_check_matrix,
)
from ._utils import complement_indices, sign_vectors, subsets


def _candidate_stats(values):
    magnitudes = np.abs(values)
    return (
        float(np.max(magnitudes, initial=0.0)),
        int(np.count_nonzero(magnitudes > 1.0)),
        int(np.count_nonzero(magnitudes < 1.0)),
    )


def _candidate_value(values, m, n):
    value, greater, less = _candidate_stats(values)
    # These are strict comparisons, as in Roth Theorem 5, equation (15).
    if greater > m or less > n - m - 1:
        return -float("inf")
    return value


def _primal_transforms(G, tol):
    k, n = G.shape
    scale = matrix_scale(G)
    for I in subsets(n, k):
        lu = factor_columns(G, I, tol, scale)
        if lu.rank == k:
            yield lu.solve(G[:, complement_indices(n, I)]).T


def h_m_roth_primal_combinatorial(G, m=None, tol=1e-10):
    """Return one height, or the shared-computation profile h_1,...,h_(n-k)."""
    G = validate_generator_matrix(G, tol)
    k, n = G.shape
    if m is not None:
        m = validate_m_for_columns(m, n)
        if m == 0:
            return 1.0
        if not generator_minimum_distance_exceeds_validated(G, m, tol):
            return float("inf")
        best = -float("inf")
        for R in _primal_transforms(G, tol):
            for signs in sign_vectors(k):
                best = max(best, _candidate_value(R @ signs, m, n))
        return best

    max_m = n - k
    if max_m == 0:
        return []
    finite_m_count = min(max_m, generator_minimum_distance_validated(G, tol) - 1)
    result = [float("inf")] * max_m
    if finite_m_count == 0:
        return result
    result[:finite_m_count] = [-float("inf")] * finite_m_count
    # Each basis solve and sign-vector product is shared across the profile.
    for R in _primal_transforms(G, tol):
        for signs in sign_vectors(k):
            value, greater, less = _candidate_stats(R @ signs)
            first = max(1, greater)
            last = min(finite_m_count, n - less - 1)
            for current_m in range(first, last + 1):
                result[current_m - 1] = max(result[current_m - 1], value)
    return result


def _pruned_sign_maximum(R, m, n):
    # Explore the most influential coordinates first; sign reversal allows
    # fixing the first sign after this permutation as well.
    order = np.argsort(-np.sum(np.abs(R), axis=0), kind="stable")
    R = R[:, order]
    k = R.shape[1]
    suffix_abs = np.zeros((R.shape[0], k + 1))
    for col in range(k - 1, -1, -1):
        suffix_abs[:, col] = suffix_abs[:, col + 1] + np.abs(R[:, col])
    partial = R[:, 0].copy()
    best = -float("inf")

    def visit(depth):
        nonlocal best
        upper_bound = float(np.max(np.abs(partial) + suffix_abs[:, depth], initial=0.0))
        if upper_bound <= best:
            return
        if depth == k:
            best = max(best, _candidate_value(partial, m, n))
            return
        partial[:] += R[:, depth]
        visit(depth + 1)
        partial[:] -= 2.0 * R[:, depth]
        visit(depth + 1)
        partial[:] += R[:, depth]

    visit(1)
    return best


def h_m_roth_primal_combinatorial_pruning(G, m, tol=1e-10):
    """Evaluate the primal formula with an exact sign-search upper bound."""
    G = validate_generator_matrix(G, tol)
    m = validate_m_for_columns(m, G.shape[1])
    if m == 0:
        return 1.0
    if not generator_minimum_distance_exceeds_validated(G, m, tol):
        return float("inf")
    return max(_pruned_sign_maximum(R, m, G.shape[1]) for R in _primal_transforms(G, tol))


def h_m_roth_dual_combinatorial_generator(G, m, tol=1e-10):
    """Evaluate the dual formula from a generator matrix."""
    G = validate_generator_matrix(G, tol)
    k, n = G.shape
    m = validate_m_for_columns(m, n)
    if m == 0:
        return 1.0
    if not generator_minimum_distance_exceeds_validated(G, m, tol):
        return float("inf")
    scale = matrix_scale(G)
    best = -float("inf")
    for S in subsets(n, m):
        targets = G[:, list(S)]
        minima = np.full(m, np.inf)
        found = False
        for I in subsets(complement_indices(n, S), k):
            lu = factor_columns(G, I, tol, scale)
            if lu.rank != k:
                continue
            coefficients = lu.solve(targets)
            minima = np.minimum(minima, np.sum(np.abs(coefficients), axis=0))
            found = True
        if not found:
            raise RuntimeError("No invertible I subset of S^c was found.")
        # Each target has its own minimum: max_i min_I, not min_I max_i.
        best = max(best, float(np.max(minima)))
    return best


def h_m_roth_dual_combinatorial_parity(H, m, tol=1e-10):
    """Evaluate the dual formula from a parity-check matrix."""
    H = validate_parity_check_matrix(H, tol)
    r, n = H.shape
    m = validate_m_for_columns(m, n)
    if m == 0:
        return 1.0
    if not parity_check_minimum_distance_exceeds_validated(H, m, tol):
        return float("inf")
    scale = matrix_scale(H)
    best = -float("inf")
    for S in subsets(n, m):
        minima = np.full(m, np.inf)
        found = False
        for extra in subsets(complement_indices(n, S), r - m):
            J = sorted((*S, *extra))
            lu = factor_columns(H, J, tol, scale)
            if lu.rank != r:
                continue
            coefficients = lu.solve(H[:, complement_indices(n, J)])
            positions = np.searchsorted(J, S)
            row_norms = np.sum(np.abs(coefficients[positions]), axis=1)
            minima = np.minimum(minima, row_norms)
            found = True
        if not found:
            raise RuntimeError("No invertible J superset of S was found.")
        best = max(best, float(np.max(minima)))
    return best
