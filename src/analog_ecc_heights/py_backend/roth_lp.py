"""Roth's primal and dual LP enumerations using SciPy."""

from itertools import combinations
from math import inf

import numpy as np

from ._lp_solver import solve_lp


def _roth_lp(G, m, early_quit_threshold, *, primal):
    if m == 0:
        return float(min(1.0, early_quit_threshold))
    k, n = G.shape
    q = n - m
    A = np.empty((2 * q, k) if primal else (k, 2 * q))
    ones = np.ones(2 * q)
    best = -inf

    for S in combinations(range(n), m):
        selected = set(S)
        complement = [j for j in range(n) if j not in selected]
        if primal:
            # -1 <= u*g_j <= 1 for every j outside S.
            A[::2] = G[:, complement].T
            A[1::2] = -A[::2]
        else:
            # G_complement * (y-z) = g_i, with y,z >= 0.
            A[:, :q] = G[:, complement]
            A[:, q:] = -A[:, :q]
        for i in S:
            if primal:
                result = solve_lp(G[:, i], A_ub=A, b_ub=ones, maximize=True)
            else:
                result = solve_lp(ones, A_eq=A, b_eq=G[:, i], nonnegative=True)
            if result.status == "optimal":
                best = max(best, result.value)
            elif (primal and result.status == "unbounded") or (
                not primal and result.status == "infeasible"
            ):
                return float(early_quit_threshold)
            else:
                raise RuntimeError(f"Unexpected {result.status} Roth {'primal' if primal else 'dual'} LP.")
            # Only a completed inner minimization certifies a dual value.
            if best > early_quit_threshold:
                return float(early_quit_threshold)
    return float(min(best, early_quit_threshold))


def h_m_roth_primal_lp(G, m, early_quit_threshold=inf):
    """Evaluate Roth's primal LP enumeration, with an optional height cap."""
    return _roth_lp(G, m, early_quit_threshold, primal=True)


def h_m_roth_dual_lp(G, m, early_quit_threshold=inf):
    """Evaluate Roth's dual LP enumeration, with an optional height cap."""
    return _roth_lp(G, m, early_quit_threshold, primal=False)
