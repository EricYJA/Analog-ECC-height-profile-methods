"""Combinatorial methods using generator or parity-check matrices."""

from . import _validation as validate
from ._dispatch import run


def _solve(method, matrix, m, backend, tol, num_threads):
    parity = method.endswith("_parity")
    matrix = validate.matrix(matrix, "H" if parity else "G", parity=parity)
    rows, n = matrix.shape
    if rows > n or (parity and rows == n):
        raise ValueError("H must have rows < columns." if parity else "G must have rows <= columns.")
    profile = method == "h_m_roth_primal_combinatorial"
    m = validate.height_index(m, n, allow_none=profile)
    tol = validate.tolerance(tol)
    if "_mds_" in method and m != (rows if parity else n - rows):
        raise ValueError("MDS methods require m equal to the code redundancy.")
    result = run(method, matrix, m, backend=backend, num_threads=num_threads, tol=tol)
    return [float(x) for x in result] if m is None else float(result)


def h_m_roth_primal_combinatorial(G, m=None, *, backend="python", tol=1e-10, num_threads=None):
    """Return scalar h_m, or [h_1, ..., h_(d-1)] when m is None.

    G must have full row rank. Scalar indices satisfy 0 <= m < n.
    The finite profile uses minimum distance d under tol; d=1 returns [].
    It shares work across heights. Backends: python, cpp.
    """
    return _solve("h_m_roth_primal_combinatorial", G, m, backend, tol, num_threads)


def h_m_roth_primal_combinatorial_pruning(G, m, *, backend="python", tol=1e-10, num_threads=None):
    """Compute a scalar primal combinatorial height with pruning."""
    return _solve("h_m_roth_primal_combinatorial_pruning", G, m, backend, tol, num_threads)


def h_m_roth_dual_combinatorial_generator(G, m, *, backend="python", tol=1e-10, num_threads=None):
    """Compute a dual combinatorial height from a full-row-rank generator G."""
    return _solve("h_m_roth_dual_combinatorial_generator", G, m, backend, tol, num_threads)


def h_m_roth_dual_combinatorial_parity(H, m, *, backend="python", tol=1e-10, num_threads=None):
    """Compute a dual combinatorial height from a full-row-rank parity matrix H."""
    return _solve("h_m_roth_dual_combinatorial_parity", H, m, backend, tol, num_threads)


def h_m_roth_mds_combinatorial(G, m, *, backend="python", tol=1e-10, num_threads=None):
    """Compute the MDS specialization with m = n-k; reject non-MDS codes.

    For zero redundancy, preserves the native specialization's return of 0.0.
    """
    return _solve("h_m_roth_mds_combinatorial", G, m, backend, tol, num_threads)


def h_m_roth_mds_combinatorial_parity(H, m, *, backend="python", tol=1e-10, num_threads=None):
    """Compute the MDS specialization with m = H.shape[0]; reject non-MDS codes.

    For zero redundancy, preserves the native specialization's return of 0.0.
    """
    return _solve("h_m_roth_mds_combinatorial_parity", H, m, backend, tol, num_threads)
