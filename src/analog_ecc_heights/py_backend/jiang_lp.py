"""Jiang's original and simplified enumerations using SciPy linear programs."""

from itertools import combinations
from math import inf

import numpy as np

from ._lp_solver import solve_lp


def h_m_jiang_original_lp(G, m):
    """Evaluate the original sign/order LP enumeration for a validated G, m."""
    k, n = G.shape
    rows = 2 * (m - 1) + 2 * (n - m - 1)
    A = np.empty((rows, k))
    upper = np.empty(rows)
    best = 0.0

    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            remaining = tuple(j for j in range(n) if j != a and j != b)
            for X in combinations(remaining, m - 1):
                selected = set(X)
                Y = tuple(j for j in remaining if j not in selected)
                for mask in range(1 << m):
                    sa = 1.0 if mask & 1 else -1.0
                    cost = sa * G[:, a]
                    row = 0
                    for t, x in enumerate(X):
                        sx = 1.0 if mask & (1 << (t + 1)) else -1.0
                        # 1 <= sx*c_x <= sa*c_a.
                        A[row] = sx * G[:, x] - cost
                        upper[row] = 0.0
                        A[row + 1] = -sx * G[:, x]
                        upper[row + 1] = -1.0
                        row += 2
                    for y in Y:
                        A[row] = G[:, y]
                        A[row + 1] = -G[:, y]
                        upper[row : row + 2] = 1.0
                        row += 2
                    result = solve_lp(
                        cost,
                        A_ub=A if rows else None,
                        b_ub=upper if rows else None,
                        A_eq=G[:, b].reshape(1, k),
                        b_eq=np.ones(1),
                        maximize=True,
                    )
                    if result.status == "optimal":
                        best = max(best, result.value)
                    elif result.status == "unbounded":
                        return inf
                    # Infeasible sign/order cases make no contribution.
    return float(best)


def h_m_jiang_simplified_lp(G, m, early_quit_threshold=inf):
    """Evaluate the simplified enumeration, capped at early_quit_threshold."""
    k, n = G.shape
    A = np.empty((m - 1 + 2 * (n - m), k))
    upper = np.ones(A.shape[0])
    upper[: m - 1] = 0.0
    best = -inf

    for a in range(n):
        cost = G[:, a]
        remaining = tuple(j for j in range(n) if j != a)
        for X in combinations(remaining, m - 1):
            selected = set(X)
            row = 0
            for x in X:
                A[row] = G[:, x] - cost
                row += 1
            for y in remaining:
                if y in selected:
                    continue
                A[row] = G[:, y]
                A[row + 1] = -G[:, y]
                row += 2
            result = solve_lp(cost, A_ub=A, b_ub=upper, maximize=True)
            if result.status == "optimal":
                best = max(best, result.value)
            elif result.status == "unbounded":
                return float(early_quit_threshold)
            else:
                raise RuntimeError("Simplified Jiang LP is infeasible although u=0 is feasible.")
            if best > early_quit_threshold:
                return float(early_quit_threshold)
    return float(min(best, early_quit_threshold))
