"""Roth's MDS specialization for generator and parity-check matrices."""

import numpy as np

from ._linalg import (
    factor_columns,
    generator_minimum_distance_exceeds_validated,
    matrix_scale,
    parity_check_minimum_distance_exceeds_validated,
    validate_generator_matrix,
    validate_m_for_columns,
    validate_parity_check_matrix,
)
from ._utils import complement_indices, subsets


def _mds_height(matrix, m, tol, parity):
    matrix = (validate_parity_check_matrix if parity else validate_generator_matrix)(matrix, tol)
    n = matrix.shape[1]
    m = validate_m_for_columns(m, n)
    redundancy = matrix.shape[0] if parity else n - matrix.shape[0]
    if m != redundancy:
        raise ValueError("MDS form requires m equal to the code redundancy.")
    # Zero redundancy describes the full space, whose h_0 is one.
    if redundancy == 0:
        return 1.0
    distance_exceeds = (parity_check_minimum_distance_exceeds_validated if parity
                        else generator_minimum_distance_exceeds_validated)
    if not distance_exceeds(matrix, redundancy, tol):
        raise ValueError("The matrix must define an MDS code.")
    scale = matrix_scale(matrix)
    best = -float("inf")
    for S in subsets(n, redundancy):
        Sc = complement_indices(n, S)
        basis, targets = (S, Sc) if parity else (Sc, S)
        lu = factor_columns(matrix, basis, tol, scale)
        coefficients = lu.solve(matrix[:, list(targets)])
        value = float(np.max(np.sum(np.abs(coefficients), axis=1 if parity else 0)))
        best = max(best, value)
    return best


def h_m_roth_mds_combinatorial(G, m, tol=1e-10):
    """Return the redundancy-height of an MDS generator matrix."""
    return _mds_height(G, m, tol, False)


def h_m_roth_mds_combinatorial_parity(H, m, tol=1e-10):
    """Return the redundancy-height of an MDS parity-check matrix."""
    return _mds_height(H, m, tol, True)
