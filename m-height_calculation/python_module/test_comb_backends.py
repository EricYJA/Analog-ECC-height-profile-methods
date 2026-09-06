"""Combinatorial API consolidation and MDS parity regression tests."""
import unittest

import numpy as np
import solve_m_height_cpp as backend


class CombinatorialTests(unittest.TestCase):
    generator_names = (
        "h_m_roth_primal_combinatorial",
        "h_m_roth_primal_combinatorial_pruning",
        "h_m_roth_dual_combinatorial_generator",
    )
    parity_name = "h_m_roth_dual_combinatorial_parity"
    mds_g_name = "h_m_roth_mds_combinatorial"
    mds_h_name = "h_m_roth_mds_combinatorial_parity"
    all_name = "h_m_roth_primal_combinatorial"

    @staticmethod
    def systematic_pair(p):
        k, r = p.shape
        return np.column_stack((np.eye(k), p)), np.column_stack((-p.T, np.eye(r)))

    def test_exports(self):
        for name in (*self.generator_names, self.parity_name,
                     self.mds_g_name, self.mds_h_name, self.all_name):
            self.assertTrue(hasattr(backend, name), name)
        for name in (*self.generator_names, self.parity_name, self.mds_g_name, self.mds_h_name):
            self.assertFalse(hasattr(backend, name + "_omp"), name)
        self.assertFalse(hasattr(backend, "h_m_roth_primal_combinatorial_omp_all"))
        self.assertFalse(hasattr(backend, "h_m_roth_primal_combinatorial_all"))

    def test_optional_m_dispatch(self):
        fn = backend.h_m_roth_primal_combinatorial
        g, _ = self.systematic_pair(np.array([[1., 1.], [1., 2.]]))
        for threads in (1, 16):
            profile = fn(g, num_threads=threads)
            self.assertIsInstance(profile, list)
            self.assertEqual(profile, fn(g, m=None, num_threads=threads))
            self.assertEqual(profile, fn(g, None, 1e-10, threads))
            self.assertEqual(len(profile), g.shape[1] - g.shape[0])
            for m, expected in enumerate(profile, 1):
                scalar = fn(g, m, 1e-10, threads)
                self.assertIsInstance(scalar, float)
                self.assertAlmostEqual(scalar, expected)
            self.assertEqual(fn(g, m=0, num_threads=threads), 1.)
            self.assertEqual(fn(np.eye(2), m=None, num_threads=threads), [])
            with self.assertRaises(ValueError):
                fn(g, m=-1, num_threads=threads)
            with self.assertRaises(ValueError):
                fn(g, m=None, tol=-1., num_threads=threads)
        with self.assertRaises(ValueError):
            fn(g, m=None, num_threads=0)
        with self.assertRaises(TypeError):
            fn(g, m="all")

    def test_general_methods_threads_and_profiles(self):
        # Existing native initial, distance-2, MDS, and complement cases.
        for p in (np.array([[1.], [1.]]),
                  np.eye(2),
                  np.array([[1., 1.], [1., 2.]]),
                  np.array([[1., 2., -1.], [2., -1., 3.]])):
            g, h = self.systematic_pair(p)
            n, r = g.shape[1], h.shape[0]
            expected_profile = []
            for m in range(n):
                expected = backend.h_m_roth_primal_lp_glpk(g, m, float("inf"))
                if 1 <= m <= r:
                    expected_profile.append(expected)
                methods = [(name, g) for name in self.generator_names]
                methods.append((self.parity_name, h))
                for threads in (1, 2, 16):
                    for name, matrix in methods:
                        with self.subTest(name=name, m=m, threads=threads, p=p.tolist()):
                            value = getattr(backend, name)(matrix, m, num_threads=threads)
                            np.testing.assert_allclose(value, expected, rtol=1e-9, atol=1e-9)
            for threads in (1, 2, 16):
                np.testing.assert_allclose(
                    getattr(backend, self.all_name)(g, num_threads=threads),
                    expected_profile, rtol=1e-9, atol=1e-9)

    def test_mds_parity_against_generator_and_lp(self):
        for p in (np.array([[1.], [1.]]),
                  np.array([[1., 1.], [1., 2.]]),
                  np.array([[1., 2., -1.], [2., -1., 3.]]),
                  np.array([[1., 2., 3.]])):
            g, h = self.systematic_pair(p)
            r = h.shape[0]
            expected = backend.h_m_roth_primal_lp_glpk(g, r, float("inf"))
            for threads in (1, 2, 16):
                for name, matrix in ((self.mds_g_name, g), (self.mds_h_name, h)):
                    with self.subTest(name=name, threads=threads, p=p.tolist()):
                        np.testing.assert_allclose(
                            getattr(backend, name)(matrix, r, num_threads=threads),
                            expected, rtol=1e-9, atol=1e-9)

    def test_dual_minimum_is_per_target(self):
        # max_i min_I = 3, whereas the incorrect min_I max_i gives 10/3.
        g, h = self.systematic_pair(np.array([[1., 3., 3., -2.],
                                             [-2., 1., 3., -2.]]))
        for threads in (1, 16):
            for name, matrix in ((self.generator_names[2], g), (self.parity_name, h)):
                with self.subTest(name=name, threads=threads):
                    self.assertAlmostEqual(
                        getattr(backend, name)(matrix, 2, num_threads=threads), 3.)

    def test_batched_solves_against_lp(self):
        rng = np.random.default_rng(714)
        for k, r in ((1, 3), (2, 2), (2, 3), (3, 2), (3, 3)):
            g, h = self.systematic_pair(rng.uniform(-2., 2., size=(k, r)))
            expected = [backend.h_m_roth_primal_lp_glpk(g, m, float("inf"))
                        for m in range(1, r + 1)]
            for scale in (1e-4, 1., 1e4):
                for threads in (1, 16):
                    with self.subTest(k=k, r=r, scale=scale, threads=threads):
                        np.testing.assert_allclose(
                            getattr(backend, self.all_name)(g * scale, num_threads=threads),
                            expected, rtol=1e-8, atol=1e-9)
                        for m, reference in enumerate(expected, 1):
                            methods = [(name, g * scale) for name in self.generator_names]
                            methods.append((self.parity_name, h / scale))
                            if m == r:
                                methods += [(self.mds_g_name, g * scale),
                                            (self.mds_h_name, h / scale)]
                            for name, matrix in methods:
                                with self.subTest(name=name, m=m):
                                    np.testing.assert_allclose(
                                        getattr(backend, name)(matrix, m, num_threads=threads),
                                        reference, rtol=1e-8, atol=1e-9)

    def test_validation_and_zero_redundancy(self):
        g, h = self.systematic_pair(np.array([[1., 1.], [1., 2.]]))
        methods = [(name, g) for name in (*self.generator_names, self.mds_g_name)]
        methods += [(self.parity_name, h), (self.mds_h_name, h)]
        for name, matrix in methods:
            fn = getattr(backend, name)
            for threads in (0, -1):
                with self.subTest(name=name, threads=threads):
                    with self.assertRaises(ValueError):
                        fn(matrix, 2, num_threads=threads)
            for tol in (-1., float("nan")):
                with self.assertRaises(ValueError):
                    fn(matrix, 2, tol=tol)
        for threads in (0, -1):
            with self.assertRaises(ValueError):
                getattr(backend, self.all_name)(g, num_threads=threads)
        bad_g, bad_h = self.systematic_pair(np.eye(2))
        for threads in (1, 16):
            for name, matrix in ((self.mds_g_name, bad_g), (self.mds_h_name, bad_h)):
                with self.assertRaises(ValueError):
                    getattr(backend, name)(matrix, 2, num_threads=threads)
            for name, matrix in ((self.mds_g_name, g), (self.mds_h_name, h)):
                with self.assertRaises(ValueError):
                    getattr(backend, name)(matrix, 1, num_threads=threads)
            for name, matrix in ((self.mds_g_name, np.eye(3)),
                                 (self.mds_h_name, np.empty((0, 3)))):
                self.assertEqual(getattr(backend, name)(matrix, 0, num_threads=threads), 1.)
        with self.assertRaises(ValueError):
            getattr(backend, self.mds_h_name)(np.ones((2, 4)), 2)
        with self.assertRaises(ValueError):
            getattr(backend, self.mds_h_name)(np.array([[1., np.nan]]), 1)


if __name__ == "__main__":
    unittest.main()
