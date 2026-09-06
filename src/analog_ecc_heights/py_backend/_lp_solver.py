"""Small SciPy LP adapter shared by the Python height algorithms."""

from typing import NamedTuple

import numpy as np
from scipy.optimize import linprog


class LpResult(NamedTuple):
    status: str
    value: float = float("nan")


def solve_lp(
    cost,
    *,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    maximize=False,
    nonnegative=False,
):
    """Solve an LP, separating infeasibility/unboundedness from solver errors.

    The default variable bounds are free, as required by generator-coordinate
    LPs. Dual LPs opt into nonnegative variables explicitly. A failed solve
    never supplies an objective value to an outer height maximization.
    """
    if A_eq is not None and b_eq is not None:
        # An exactly zero equality row cannot equal a nonzero right-hand
        # side. Older HiGHS versions can report an unknown status for this
        # contradiction with presolve off; certify it directly, without a
        # numerical rank tolerance or a solver fallback.
        zero_rows = np.all(np.asarray(A_eq) == 0, axis=1)
        if np.any(zero_rows & (np.asarray(b_eq) != 0)):
            return LpResult("infeasible")
    result = linprog(
        -cost if maximize else cost,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None) if nonnegative else (None, None),
        method="highs",
        # Presolve can misclassify feasible, unbounded homogeneous LPs as
        # infeasible. The native HiGHS implementation disables it as well.
        options={"presolve": False},
    )
    if result.status == 0:
        if result.fun is None or not np.isfinite(result.fun):
            raise RuntimeError("SciPy LP returned a nonfinite optimal objective.")
        return LpResult("optimal", float(-result.fun if maximize else result.fun))
    if result.status == 2:
        # Some SciPy releases map HiGHS model errors to status 2 as well.
        # Only an explicitly identified infeasible model has mathematical
        # meaning for the caller; rejected models must raise an error.
        if "infeasible" in str(result.message).lower():
            return LpResult("infeasible")
    elif result.status == 3:
        return LpResult("unbounded")
    raise RuntimeError(f"SciPy LP failed (status {result.status}): {result.message}")
