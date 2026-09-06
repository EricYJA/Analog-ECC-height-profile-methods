"""Run with PYTHONPATH pointing to the module build to test."""
import math
import os
import subprocess
import sys
import unittest
import numpy as np
try:
    from analog_ecc_heights.cpp_backend.adapter import load_native
    s = load_native()
except ImportError as exc:
    raise unittest.SkipTest(f"Native extension unavailable: {exc}") from exc


class LpBackendTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(s, "h_m_roth_primal_lp_highs"), "HiGHS not built")
    def test_highs_thread_counts_and_layouts(self):
        G = np.array([[1., 0., 1., 2., -1.], [0., 1., 2., -1., 3.]])
        padded = np.zeros((2, 10))
        padded[:, ::2] = G
        readonly = G.copy()
        readonly.flags.writeable = False
        for matrix in (G, np.asfortranarray(G), padded[:, ::2], G.astype(np.float32), readonly):
            for threads in (1, 2, 4, 16):
                for m in (1, 2, 3):
                    expected = s.h_m_roth_primal_combinatorial(G, m)
                    with self.subTest(threads=threads, m=m, strides=matrix.strides):
                        self.assert_value(s.h_m_jiang_original_lp_highs(matrix, m, num_threads=threads),
                                          expected)
                        for fn in (s.h_m_jiang_simplified_lp_highs_early_quit,
                                   s.h_m_roth_primal_lp_highs, s.h_m_roth_dual_lp_highs):
                            for threshold in (1.5, math.inf):
                                self.assert_value(fn(matrix, m, threshold, num_threads=threads),
                                                  min(expected, threshold))

    @unittest.skipUnless(hasattr(s, "h_m_roth_primal_lp_highs"), "HiGHS not built")
    def test_highs_validation_and_worker_errors(self):
        G = np.array([[1., 0., 1.], [0., 1., 1.]])
        methods = (
            lambda matrix, threads: s.h_m_jiang_original_lp_highs(matrix, 1, threads),
            lambda matrix, threads: s.h_m_jiang_simplified_lp_highs_early_quit(matrix, 1, math.inf, threads),
            lambda matrix, threads: s.h_m_roth_primal_lp_highs(matrix, 1, math.inf, threads),
            lambda matrix, threads: s.h_m_roth_dual_lp_highs(matrix, 1, math.inf, threads),
        )
        for fn in methods:
            for threads in (0, -1):
                with self.assertRaises(ValueError):
                    fn(G, threads)
            # Oversized finite coefficients cause model rejection inside the
            # workers; errors must propagate without crossing the OMP boundary.
            with self.assertRaises(RuntimeError):
                fn(G * 1e30, 4)

    def backends(self):
        return ["glpk"] + (["highs"] if hasattr(s, "h_m_roth_primal_lp_highs") else [])

    def roth_methods(self):
        return [getattr(s, f"h_m_roth_{kind}_lp_{backend}")
                for backend in self.backends() for kind in ("primal", "dual")]

    def assert_value(self, actual, expected):
        if math.isinf(expected):
            self.assertEqual(actual, expected)
        else:
            self.assertAlmostEqual(actual, expected, delta=1e-7 * max(1., abs(expected)))

    def test_exports(self):
        expected = {f"h_m_roth_{kind}_lp_{backend}"
                    for backend in self.backends() for kind in ("primal", "dual")}
        self.assertEqual({x for x in dir(s) if x.startswith(("h_m_roth_primal_lp", "h_m_roth_dual_lp"))},
                         expected)

    def test_finite_profiles_and_thresholds(self):
        rng = np.random.default_rng(2026)
        matrices = [np.array([[1., 0., 1., 2., -1.], [0., 1., 2., -1., 3.]]),
                    np.array([[1., -2., 4.]])]
        matrices += [rng.normal(size=(2, 5)) for _ in range(5)]
        for G in matrices:
            for m in range(1, G.shape[1] - G.shape[0] + 1):
                expected = s.h_m_roth_primal_combinatorial(G, m)
                for fn in self.roth_methods() + [s.h_m_jiang_simplified_lp_glpk_early_quit]:
                    for threshold in (0.5, expected / 2, expected, expected * 2, math.inf):
                        with self.subTest(fn=fn.__name__, m=m, threshold=threshold):
                            self.assert_value(fn(G=G, m=m, early_quit_threshold=threshold),
                                              min(expected, threshold))

    def test_unbounded_and_zero_height_index(self):
        G = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
        for fn in self.roth_methods():
            for m, expected in ((0, 1.), (1, 1.), (2, math.inf), (3, math.inf)):
                for threshold in (0.5, 2., math.inf):
                    with self.subTest(fn=fn.__name__, m=m, threshold=threshold):
                        self.assert_value(fn(G, m, threshold), min(expected, threshold))

    def test_original_workspace_reuse(self):
        # Repeated changes of objective, sign constraints and normalization row
        # must not leave stale values in the reused HiGHS buffers.
        matrices = [np.array([[1., -2., 4.]]),
                    np.array([[1., 0., 1., 2.], [0., 1., -1., 3.]]),
                    np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])]
        for G in matrices:
            for m in range(1, G.shape[1]):
                expected = s.h_m_roth_primal_combinatorial(G, m)
                self.assert_value(s.h_m_jiang_simplified_lp_glpk_early_quit(G, m, math.inf),
                                  expected)
                for backend in self.backends():
                    fn = getattr(s, f"h_m_jiang_original_lp_{backend}")
                    with self.subTest(fn=fn.__name__, m=m):
                        self.assert_value(fn(G, m), expected)

    def test_interleaved_calls(self):
        G = np.array([[1., 0., 1.], [0., 1., 1.]])
        for backend in self.backends():
            simplified = getattr(s, f"h_m_jiang_simplified_lp_{backend}_early_quit")
            original = getattr(s, f"h_m_jiang_original_lp_{backend}")
            for _ in range(2):
                self.assert_value(simplified(G, 1, math.inf), 2.)
                self.assert_value(original(G, 1), 2.)
                for fn in self.roth_methods():
                    self.assert_value(fn(G, 1, 1.5), 1.5)
                    self.assert_value(fn(G, 1, math.inf), 2.)

    def test_invalid_inputs(self):
        G = np.array([[1., 0., 1.], [0., 1., 1.]])
        for fn in self.roth_methods():
            for m, threshold in ((-1, math.inf), (3, math.inf), (1, math.nan)):
                with self.assertRaises(ValueError):
                    fn(G, m, threshold)
            with self.assertRaises(ValueError):
                fn(np.array([[math.nan, 1.]]), 1, math.inf)

    def test_glpk_incremental_early_exit(self):
        # A full task list here has hundreds of millions of entries, but the
        # first LP already exceeds the threshold. Run in a bounded subprocess.
        subprocess.run(
            [sys.executable, "-c", """
import resource
import numpy as np
import solve_m_height_cpp as s
with open('/proc/self/statm') as f:
    pages = int(f.read().split()[0])
import os
limit = pages * os.sysconf('SC_PAGE_SIZE') + 256 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
assert s.h_m_jiang_simplified_lp_glpk_early_quit(np.ones((1, 28)), 14, 0.5) == 0.5
"""],
            env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"},
            check=True, capture_output=True, text=True, timeout=10,
        )

    def test_glpk_simplified_validation_and_unbounded(self):
        fn = s.h_m_jiang_simplified_lp_glpk_early_quit
        G = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
        for threshold in (0.5, 2., math.inf):
            self.assert_value(fn(G, 2, threshold), min(math.inf, threshold))
        for m, threshold in ((0, math.inf), (4, math.inf), (5, math.inf), (1, math.nan)):
            with self.assertRaises(ValueError):
                fn(G, m, threshold)
        with self.assertRaises(ValueError):
            fn(np.array([[math.nan, 1.]]), 1, math.inf)


if __name__ == "__main__":
    unittest.main()
