"""Known-answer and solver-failure checks for the independent Python LPs."""

import math
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analog_ecc_heights.py_backend import _lp_solver
from analog_ecc_heights.py_backend.jiang_lp import (
    h_m_jiang_original_lp,
    h_m_jiang_simplified_lp,
)
from analog_ecc_heights.py_backend.roth_lp import (
    h_m_roth_dual_lp,
    h_m_roth_primal_lp,
)


CAPPED_METHODS = (
    h_m_jiang_simplified_lp,
    h_m_roth_primal_lp,
    h_m_roth_dual_lp,
)
ALL_METHODS = (h_m_jiang_original_lp,) + CAPPED_METHODS


class PythonLpTests(unittest.TestCase):
    def assert_height(self, actual, expected):
        self.assertIsInstance(actual, float)
        if math.isinf(expected):
            self.assertEqual(actual, expected)
        else:
            self.assertAlmostEqual(actual, expected, delta=1e-7 * max(1.0, abs(expected)))

    def test_rank_one_known_answers_and_signed_coordinates(self):
        # Every nonzero codeword has the same ordered magnitudes 4, 2, 1.
        for G in (np.array([[1.0, -2.0, 4.0]]), np.array([[-1.0, -2.0, -4.0]])):
            for m, expected in ((1, 2.0), (2, 4.0)):
                for fn in ALL_METHODS:
                    with self.subTest(fn=fn.__name__, m=m, G=G):
                        self.assert_height(fn(G, m), expected)

    def test_triangle_code_known_answers(self):
        # c=(u,v,u+v); the largest magnitude is at most twice the second.
        G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        for fn in ALL_METHODS:
            with self.subTest(fn=fn.__name__):
                self.assert_height(fn(G, 1), 2.0)
                self.assert_height(fn(G, 2), math.inf)

    def test_unbounded_polytope_is_not_misclassified_by_presolve(self):
        # In SciPy 1.15, presolve incorrectly labels the first Roth primal
        # polytope infeasible, even though u=0 satisfies every inequality.
        # A 3-dimensional code with only one constrained coordinate has an
        # unbounded height at m=4. All formulations must agree on infinity.
        rng = np.random.default_rng(714)
        G = np.column_stack((np.eye(3), rng.uniform(-2.0, 2.0, (3, 2))))
        for fn in ALL_METHODS:
            with self.subTest(fn=fn.__name__):
                self.assert_height(fn(G, 4), math.inf)

    def test_two_columns_and_original_without_inequalities(self):
        # Original Jiang has only its equality constraint when n=2, m=1.
        G = np.array([[2.0, -3.0]])
        for fn in ALL_METHODS:
            self.assert_height(fn(G, 1), 1.5)

    def test_roth_dual_exact_zero_equality_contradictions(self):
        # The selected objective cannot be represented by the complementary
        # columns. SciPy 1.9 can return an unknown status for the zero row in
        # the dual equality, but the mathematical height is positive infinity.
        matrices = (
            np.array([[0.0, 1.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        )
        for G in matrices:
            for threshold in (0.5, 2.0, math.inf):
                with self.subTest(G=G, threshold=threshold):
                    self.assert_height(h_m_roth_dual_lp(G, 2, threshold), threshold)

    def test_finite_and_infinite_threshold_caps(self):
        G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        for fn in CAPPED_METHODS:
            for m, expected in ((1, 2.0), (2, math.inf)):
                for threshold in (-math.inf, -1.0, 0.0, 0.5, 2.0, 4.0, math.inf):
                    with self.subTest(fn=fn.__name__, m=m, threshold=threshold):
                        self.assert_height(fn(G, m, threshold), min(expected, threshold))

    def test_roth_zero_index_and_zero_columns(self):
        G = np.array([[0.0, 1.0, 0.0, 1.0]])
        for fn in (h_m_roth_primal_lp, h_m_roth_dual_lp):
            for threshold in (0.5, 2.0, math.inf):
                self.assert_height(fn(G, 0, threshold), min(1.0, threshold))
            self.assert_height(fn(G, 1), 1.0)
            self.assert_height(fn(G, 2), math.inf)
        for fn in CAPPED_METHODS:
            self.assert_height(fn(np.zeros((2, 3)), 1), 0.0)

    def test_primal_dual_agreement_for_random_small_codes(self):
        rng = np.random.default_rng(2026)
        for _ in range(3):
            G = rng.normal(size=(2, 4))
            for m in (1, 2):
                expected = h_m_roth_primal_lp(G, m)
                for fn in (h_m_roth_dual_lp, h_m_jiang_original_lp, h_m_jiang_simplified_lp):
                    with self.subTest(fn=fn.__name__, m=m):
                        self.assert_height(fn(G, m), expected)

    def test_numerical_model_errors_are_not_heights(self):
        G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]) * 1e30
        for fn in ALL_METHODS:
            with self.subTest(fn=fn.__name__), self.assertRaisesRegex(RuntimeError, "SciPy LP failed"):
                fn(G, 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "uses Linux process memory limits")
    def test_threshold_enumeration_has_bounded_memory(self):
        # There are tens of millions of subsets, but the first LP suffices.
        # Bound subprocess memory so an eager enumeration fails safely.
        script = """
import os
import resource
import numpy as np
from analog_ecc_heights.py_backend.jiang_lp import h_m_jiang_simplified_lp
from analog_ecc_heights.py_backend.roth_lp import h_m_roth_primal_lp, h_m_roth_dual_lp
methods = (h_m_jiang_simplified_lp, h_m_roth_primal_lp, h_m_roth_dual_lp)
for fn in methods:
    assert fn(np.ones((1, 3)), 1, 0.5) == 0.5
with open('/proc/self/statm') as handle:
    pages = int(handle.read().split()[0])
limit = pages * os.sysconf('SC_PAGE_SIZE') + 128 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
for fn in methods:
    assert fn(np.ones((1, 28)), 14, 0.5) == 0.5
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class LpStatusTests(unittest.TestCase):
    def test_optimal_infeasible_and_unbounded_are_distinct(self):
        solve = _lp_solver.solve_lp
        optimum = solve(np.array([-1.0]), A_ub=np.array([[1.0]]), b_ub=np.array([2.0]))
        self.assertEqual(optimum, _lp_solver.LpResult("optimal", -2.0))
        infeasible = solve(
            np.ones(1), A_ub=np.ones((1, 1)), b_ub=-np.ones(1), nonnegative=True
        )
        self.assertEqual(infeasible.status, "infeasible")
        self.assertEqual(solve(np.ones(1)).status, "unbounded")

    def test_exact_zero_equality_row_with_nonzero_rhs_is_infeasible(self):
        for rhs in (1.0, -1.0, 1e-300):
            with self.subTest(rhs=rhs):
                result = _lp_solver.solve_lp(
                    np.ones(2), A_eq=np.zeros((1, 2)), b_eq=np.array([rhs]), nonnegative=True
                )
                self.assertEqual(result.status, "infeasible")
        # A zero equality with zero RHS is harmless and remains feasible.
        result = _lp_solver.solve_lp(
            np.ones(2), A_eq=np.zeros((1, 2)), b_eq=np.zeros(1), nonnegative=True
        )
        self.assertEqual(result, _lp_solver.LpResult("optimal", 0.0))

    def test_unfinished_solve_cannot_certify_a_height(self):
        partial = SimpleNamespace(status=1, fun=-1000.0, message="Iteration limit reached")
        G = np.array([[1.0, 2.0, 4.0]])
        with patch.object(_lp_solver, "linprog", return_value=partial):
            for fn in CAPPED_METHODS:
                with self.subTest(fn=fn.__name__), self.assertRaisesRegex(RuntimeError, "Iteration limit"):
                    fn(G, 1, 0.5)

    def test_nonfinite_optimum_is_rejected(self):
        for value in (None, math.nan, math.inf):
            result = SimpleNamespace(status=0, fun=value, message="Optimal")
            with patch.object(_lp_solver, "linprog", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                    _lp_solver.solve_lp(np.ones(1))


if __name__ == "__main__":
    unittest.main()
