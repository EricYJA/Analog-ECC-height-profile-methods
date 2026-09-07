"""Mathematical and numerical regression tests for the NumPy backend."""

from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import linprog

from analog_ecc_heights.py_backend._linalg import FullPivotLU, numerical_rank
from analog_ecc_heights.py_backend.roth_comb import (
    h_m_roth_dual_combinatorial_generator as dual_generator,
    h_m_roth_dual_combinatorial_parity as dual_parity,
    h_m_roth_primal_combinatorial as primal,
    h_m_roth_primal_combinatorial_pruning as pruning,
)
from analog_ecc_heights.py_backend.roth_mds_comb import (
    h_m_roth_mds_combinatorial as mds_generator,
    h_m_roth_mds_combinatorial_parity as mds_parity,
)


def systematic_pair(P):
    k, r = P.shape
    return np.column_stack((np.eye(k), P)), np.column_stack((-P.T, np.eye(r)))


def reference_lp_height(G, m):
    """Independent definition: bound all but m coordinates and maximize each."""
    if m == 0:
        return 1.0
    k, n = G.shape
    best = 1.0
    for S in combinations(range(n), m):
        Sc = [i for i in range(n) if i not in S]
        bounds = G[:, Sc].T
        for i in S:
            result = linprog(-G[:, i], A_ub=np.vstack((bounds, -bounds)),
                             b_ub=np.ones(2 * len(Sc)), bounds=[(None, None)] * k,
                             method="highs")
            if result.status == 2:
                # Zero is feasible; resolve HiGHS presolve ambiguity.
                result = linprog(-G[:, i], A_ub=np.vstack((bounds, -bounds)),
                                 b_ub=np.ones(2 * len(Sc)), bounds=[(None, None)] * k,
                                 method="highs", options={"presolve": False})
            if result.status == 3:
                return float("inf")
            assert result.success, result.message
            best = max(best, -result.fun)
    return best


@pytest.mark.parametrize("P", [
    np.array([[1.], [1.]]),
    np.eye(2),
    np.array([[1., 1.], [1., 2.]]),
    np.array([[1., 2., -1.], [2., -1., 3.]]),
    np.random.default_rng(714).uniform(-2., 2., (3, 2)),
])
def test_general_formulations_against_independent_lp(P):
    G, H = systematic_pair(P)
    expected_profile = []
    for m in range(G.shape[1]):
        expected = reference_lp_height(G, m)
        for method, matrix in ((primal, G), (pruning, G),
                               (dual_generator, G), (dual_parity, H)):
            value = method(matrix, m)
            assert isinstance(value, float)
            np.testing.assert_allclose(value, expected, rtol=1e-8, atol=1e-9)
        if m >= 1 and np.isfinite(expected):
            expected_profile.append(expected)
    assert isinstance(primal(G), list)
    np.testing.assert_allclose(primal(G), expected_profile, rtol=1e-8, atol=1e-9)


def test_rank_one_known_heights_and_distance():
    G = np.array([[1., -2., 4.]])
    assert primal(G) == [2., 4.]
    for m, height in enumerate((1., 2., 4.)):
        assert primal(G, m) == height
        assert pruning(G, m) == height
        assert dual_generator(G, m) == height
    assert primal(np.array([[0., 1., 2.]])) == [2.]
    assert primal(np.array([[0., 1., 2.]]), 2) == float("inf")


def test_dual_minimum_is_per_target():
    # max_i min_I = 3; the incorrect min_I max_i equals 10/3.
    G, H = systematic_pair(np.array([[1., 3., 3., -2.], [-2., 1., 3., -2.]]))
    assert dual_generator(G, 2) == pytest.approx(3.)
    assert dual_parity(H, 2) == pytest.approx(3.)


@pytest.mark.parametrize("scale", [1e-4, 1., 1e4])
def test_scaling_and_mds_specialization(scale):
    G, H = systematic_pair(np.array([[1., 2., -1.], [2., -1., 3.]]))
    r = H.shape[0]
    expected = reference_lp_height(G, r)
    np.testing.assert_allclose(primal(G * scale), primal(G), rtol=1e-10)
    for method, matrix in ((mds_generator, G * scale), (mds_parity, H / scale)):
        assert method(matrix, r) == pytest.approx(expected)
        with pytest.raises(ValueError, match="redundancy"):
            method(matrix, r - 1)
    bad_G, bad_H = systematic_pair(np.eye(2))
    for method, matrix in ((mds_generator, bad_G), (mds_parity, bad_H)):
        with pytest.raises(ValueError, match="MDS"):
            method(matrix, 2)


def test_profile_shares_basis_solves(monkeypatch):
    import analog_ecc_heights.py_backend.roth_comb as module
    G, _ = systematic_pair(np.array([[1., 1.], [1., 2.]]))
    calls = 0
    original = module.factor_columns

    def count_factor(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(module, "factor_columns", count_factor)
    np.testing.assert_allclose(primal(G), [2., 3.])
    assert calls == 6  # Six bases, rather than six per requested height.


def test_parent_scale_controls_near_rank_distance():
    G = np.array([[1., 1e-12]])
    assert primal(G, 1, tol=1e-10) == float("inf")
    assert primal(G, 1, tol=1e-14) == pytest.approx(1e12)
    assert primal(G, tol=1e-10) == []
    assert primal(G, tol=1e-14) == pytest.approx([1e12])
    tiny = np.array([[1e-12]])
    assert numerical_rank(tiny, 1e-10) == 1
    assert numerical_rank(tiny, 1e-10, reference_scale=1.) == 0
    # The cutoff is strict, including equality at the requested tolerance.
    assert numerical_rank(np.diag([1., 1e-10]), 1e-10) == 1
    assert numerical_rank(np.diag([1., np.nextafter(1e-10, np.inf)]), 1e-10) == 2
    with pytest.raises(ValueError, match="full row rank"):
        primal(np.diag([1., 1e-12]), tol=1e-10)
    assert primal(np.diag([1., 1e-12]), tol=1e-14) == []


def test_complete_pivot_solve_handles_row_and_column_permutations():
    matrix = np.array([[0., 1., 2.], [3., 4., 9.], [1., -2., 3.]])
    targets = np.array([[1., 3.], [2., -1.], [4., 5.]])
    lu = FullPivotLU(matrix, 1e-10)
    np.testing.assert_allclose(matrix @ lu.solve(targets), targets, atol=1e-12)
    np.testing.assert_allclose(matrix @ lu.solve(targets[:, 0]), targets[:, 0], atol=1e-12)


def test_validation_and_zero_redundancy():
    assert primal(np.eye(3)) == []
    assert primal(np.eye(3), 0) == 1.
    assert primal(np.eye(3), 1) == float("inf")
    assert dual_parity(np.empty((0, 3)), 0) == 1.
    assert dual_parity(np.empty((0, 3)), 1) == float("inf")
    # Compatibility with the existing specialized C++ implementation.
    assert mds_generator(np.eye(3), 0) == 0.
    assert mds_parity(np.empty((0, 3)), 0) == 0.
    for matrix in (np.ones((2, 3)), np.empty((0, 2)), np.eye(3)[:, :2],
                   np.array([[1., np.nan]]), np.array([[1., np.inf]]),
                   np.array([[1., 1j]]), np.array([1., 2.])):
        with pytest.raises(ValueError):
            primal(matrix)
    for tol in (-1., np.nan, np.inf):
        with pytest.raises(ValueError, match="tol"):
            primal([[1., 2.]], tol=tol)
    for m in (-1, 2):
        with pytest.raises(ValueError, match="m must"):
            primal([[1., 2.]], m)
    for m in ("all", 1.0, True):
        with pytest.raises(TypeError, match="integer"):
            primal([[1., 2.]], m)
