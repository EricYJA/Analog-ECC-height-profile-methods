"""Linear algebra using the native methods' full-pivot rank convention.

Rank is determined by complete-pivot LU with an absolute cutoff of
``tol * reference_scale``. Submatrices use the *parent matrix's* largest
absolute entry as their reference scale. An SVD's default rank threshold
would change minimum-distance decisions for small or nearly singular bases.
"""

import operator

import numpy as np

from ._utils import complement_indices, subsets


def matrix_scale(matrix):
    """Largest absolute entry, including the empty-matrix convention."""
    return float(np.max(np.abs(matrix), initial=0.0))


class FullPivotLU:
    """Complete-pivot LU with one factorization for rank checks and solves."""

    def __init__(self, matrix, tol, reference_scale=None):
        self.lu = np.array(matrix, dtype=float, copy=True)
        rows, cols = self.lu.shape
        self.row_order = np.arange(rows)
        self.col_order = np.arange(cols)
        if reference_scale is None:
            reference_scale = matrix_scale(self.lu)
        diagonal_size = min(rows, cols)
        max_pivot = 0.0
        nonzero = 0
        for step in range(diagonal_size):
            corner = np.abs(self.lu[step:, step:])
            # Eigen stores MatrixXd in column-major order. Preserve its tie order.
            pivot_index = int(np.argmax(corner.ravel(order="F")))
            row = step + pivot_index % corner.shape[0]
            col = step + pivot_index // corner.shape[0]
            pivot_size = abs(self.lu[row, col])
            if pivot_size == 0.0:
                break
            max_pivot = max(max_pivot, pivot_size)
            nonzero += 1
            if row != step:
                self.lu[[step, row], :] = self.lu[[row, step], :]
                self.row_order[[step, row]] = self.row_order[[row, step]]
            if col != step:
                self.lu[:, [step, col]] = self.lu[:, [col, step]]
                self.col_order[[step, col]] = self.col_order[[col, step]]
            if step + 1 < rows:
                self.lu[step + 1:, step] /= self.lu[step, step]
            if step + 1 < diagonal_size:
                self.lu[step + 1:, step + 1:] -= np.outer(
                    self.lu[step + 1:, step], self.lu[step, step + 1:]
                )
        # Retain the multiplication order of FullPivLU::rank(), including
        # rounding exactly at the tolerance boundary.
        threshold = max_pivot * (tol * reference_scale / max_pivot) if max_pivot else 0.0
        self.rank = int(np.count_nonzero(np.abs(self.lu.diagonal()[:nonzero]) > threshold))

    def solve(self, targets):
        """Solve square nonsingular systems, retaining the rank factorization."""
        rows, cols = self.lu.shape
        if rows != cols or self.rank != rows:
            raise np.linalg.LinAlgError("A square, numerically invertible basis is required.")
        targets = np.asarray(targets, dtype=float)
        vector = targets.ndim == 1
        if vector:
            targets = targets[:, None]
        if targets.ndim != 2 or targets.shape[0] != rows:
            raise ValueError("The right-hand side must have one row per basis row.")
        result = targets[self.row_order].copy()
        # Column-oriented substitutions mirror Eigen's triangular solves.
        for col in range(rows):
            result[col + 1:] -= self.lu[col + 1:, col, None] * result[col]
        for col in range(rows - 1, -1, -1):
            result[col] /= self.lu[col, col]
            result[:col] -= self.lu[:col, col, None] * result[col]
        unpermuted = np.empty_like(result)
        unpermuted[self.col_order] = result
        return unpermuted[:, 0] if vector else unpermuted


def numerical_rank(matrix, tol, reference_scale=None):
    return FullPivotLU(matrix, tol, reference_scale).rank


def factor_columns(matrix, columns, tol, reference_scale):
    return FullPivotLU(matrix[:, list(columns)], tol, reference_scale)


def validate_rank_tolerance(tol):
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be a finite nonnegative value.")


def _as_real_matrix(matrix, name):
    raw = np.asarray(matrix)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values.")
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def validate_generator_matrix(G, tol=1e-10):
    validate_rank_tolerance(tol)
    G = _as_real_matrix(G, "G")
    k, n = G.shape
    if k == 0 or n == 0:
        raise ValueError("G must be a non-empty generator matrix.")
    if k > n:
        raise ValueError(f"G must satisfy rows <= cols, got rows={k}, cols={n}.")
    if numerical_rank(G, tol) != k:
        raise ValueError("G must have full row rank.")
    return G


def validate_parity_check_matrix(H, tol=1e-10):
    validate_rank_tolerance(tol)
    H = _as_real_matrix(H, "H")
    r, n = H.shape
    if n == 0:
        raise ValueError("H must have at least one column.")
    if r >= n:
        raise ValueError(f"H must satisfy rows < cols, got rows={r}, cols={n}.")
    if numerical_rank(H, tol) != r:
        raise ValueError("H must have full row rank.")
    return H


def validate_m_for_columns(m, n):
    if isinstance(m, (bool, np.bool_)):
        raise TypeError("m must be an integer.")
    try:
        m = operator.index(m)
    except TypeError:
        raise TypeError("m must be an integer.") from None
    if m < 0 or m >= n:
        raise ValueError("m must satisfy 0 <= m <= n-1.")
    return m


def generator_minimum_distance_exceeds_validated(G, m, tol):
    k, n = G.shape
    if n - m < k:
        return False
    scale = matrix_scale(G)
    return all(
        numerical_rank(G[:, complement_indices(n, S)], tol, scale) == k
        for S in subsets(n, m)
    )


def parity_check_minimum_distance_exceeds_validated(H, m, tol):
    if m > H.shape[0]:
        return False
    scale = matrix_scale(H)
    return all(
        numerical_rank(H[:, list(S)], tol, scale) == m
        for S in subsets(H.shape[1], m)
    )


def generator_minimum_distance_validated(G, tol):
    redundancy = G.shape[1] - G.shape[0]
    for m in range(1, redundancy + 1):
        if not generator_minimum_distance_exceeds_validated(G, m, tol):
            return m
    return redundancy + 1
