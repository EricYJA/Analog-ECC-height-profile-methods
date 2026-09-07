"""Jiang index limits and finite-profile boundaries across entry points."""

import os
import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest

import analog_ecc_heights as heights
from analog_ecc_heights.cpp_backend.adapter import load_native


@pytest.mark.parametrize("backend", ("python", "cpp-glpk", "cpp-highs"))
@pytest.mark.parametrize("m", (31, 64))
def test_jiang_large_indices(backend, m):
    if backend not in heights.available_backends():
        pytest.skip(f"{backend} is not installed")
    # A timeout catches accidental exhaustive sign enumeration after an
    # unbounded LP. The old cap rejects 31; fixed-width masks also fail at 64.
    script = r"""
import sys
import numpy as np
import analog_ecc_heights as heights
from analog_ecc_heights.cpp_backend.adapter import load_native
from analog_ecc_heights.py_backend import jiang_lp

backend, m = sys.argv[1], int(sys.argv[2])
G_unbounded = np.eye(m + 1)  # First original LP is already unbounded.
G_finite = np.ones((1, m + 1))  # Every finite height is 1.
for threads in ((1, 4) if backend == "cpp-highs" else (1,)):
    entries = [(heights.h_m_jiang_original_lp, heights.h_m_jiang_simplified_lp,
                {"backend": backend, "num_threads": threads})]
    if backend == "python":
        entries.append((jiang_lp.h_m_jiang_original_lp,
                        jiang_lp.h_m_jiang_simplified_lp, {}))
    else:
        native = load_native(backend)
        solver = backend[4:]
        entries.append((getattr(native, f"h_m_jiang_original_lp_{solver}"),
                        getattr(native, f"h_m_jiang_simplified_lp_{solver}_early_quit"),
                        {"num_threads": threads} if solver == "highs" else {}))
    for original, simplified, options in entries:
        assert original(G_unbounded, m, **options) == float("inf")
        assert simplified(G_unbounded, m, early_quit_threshold=float("inf"),
                          **options) == float("inf")
        assert simplified(G_finite, m, early_quit_threshold=0.5, **options) == 0.5
    for invalid in (0, m + 1):
        for method in (heights.h_m_jiang_original_lp, heights.h_m_jiang_simplified_lp):
            try:
                method(G_finite, invalid, backend=backend, num_threads=threads)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{method.__name__} accepted m={invalid}")
"""
    subprocess.run([sys.executable, "-c", script, backend, str(m)],
                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
                   check=True, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize("backend", ("python", "cpp"))
@pytest.mark.parametrize("G, expected", [
    (np.array([[0., 0., 1., 2.]]), [2.]),
    (np.array([[0., 1., 2., 4.]]), [2., 4.]),
    (np.array([[1., 2., 4.]]), [2., 4.]),
    (np.array([[1., 0., 0.]]), []),
    (np.eye(3), []),
], ids=("distance-two", "distance-three", "mds", "distance-one", "square"))
def test_finite_profile_boundaries(backend, G, expected):
    if backend not in heights.available_backends():
        pytest.skip(f"{backend} is not installed")
    direct = (import_module("analog_ecc_heights.py_backend.roth_comb")
              if backend == "python" else load_native(backend))
    for threads in ((1, 4) if backend == "cpp" else (1,)):
        entries = (
            (heights.h_m_roth_primal_combinatorial,
             {"backend": backend, "num_threads": threads}),
            (direct.h_m_roth_primal_combinatorial,
             {"num_threads": threads} if backend == "cpp" else {}),
        )
        for method, options in entries:
            assert method(G, **options) == pytest.approx(expected)
            assert method(G, None, **options) == pytest.approx(expected)
            for m, value in enumerate(expected, 1):
                assert method(G, m, **options) == pytest.approx(value)
            d = len(expected) + 1
            if d < G.shape[1]:
                assert method(G, d, **options) == float("inf")
