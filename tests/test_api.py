"""Public validation and isolation of optional native code."""
import json
import os
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
import analog_ecc_heights as h

G = np.array([[1., 0., 1.], [0., 1., 1.]])

@pytest.mark.parametrize("function", [h.h_m_jiang_original_lp, h.h_m_jiang_simplified_lp,
                                    h.h_m_roth_primal_lp, h.h_m_roth_dual_lp])
def test_default_known_value(function):
    result = function(G, 1)
    assert isinstance(result, float)
    assert result == pytest.approx(2.)

def test_keyword_only_and_method_specific_options():
    with pytest.raises(TypeError):
        h.h_m_roth_primal_lp(G, 1, "python")
    with pytest.raises(TypeError):
        h.h_m_jiang_original_lp(G, 1, early_quit_threshold=1.)
    with pytest.raises(TypeError):
        h.h_m_roth_dual_lp(G, None)
    with pytest.raises(TypeError):
        h.h_m_roth_primal_combinatorial_pruning(G, None)
    with pytest.raises(ValueError, match="zero column"):
        h.h_m_jiang_original_lp(np.column_stack([G, np.zeros(2)]), 1)
    with pytest.raises(ValueError):
        h.h_m_jiang_simplified_lp(G, 0)
    with pytest.raises(ValueError):
        h.h_m_roth_primal_lp(G, 1, early_quit_threshold=np.nan)
    assert h.h_m_roth_dual_lp(G, 0, early_quit_threshold=.5) == .5

@pytest.mark.parametrize("backend", ["glpk", "highs", "auto", "cpp", None, []])
def test_invalid_lp_backends(backend):
    with pytest.raises(ValueError, match="Unsupported backend"):
        h.h_m_roth_primal_lp(G, 1, backend=backend)

@pytest.mark.parametrize("backend", ["cpp-glpk", "cpp-highs"])
def test_lp_backend_invalid_for_combinatorial(backend):
    with pytest.raises(ValueError, match="Unsupported backend"):
        h.h_m_roth_primal_combinatorial(G, 1, backend=backend)

@pytest.mark.parametrize("m", [1.5, "1", True, np.bool_(True)])
def test_integer_indices(m):
    with pytest.raises(TypeError):
        h.h_m_roth_primal_lp(G, m)

@pytest.mark.parametrize("matrix", [[1, 2], np.empty((0, 3)), np.empty((2, 0)),
                                   [[1., np.nan]], [[np.inf, 1.]], [[1j, 2.]]])
def test_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        h.h_m_roth_primal_lp(matrix, 1)

@pytest.mark.parametrize("threads,exception", [(0, ValueError), (-1, ValueError),
                                              (2, ValueError), (1.5, TypeError), (True, TypeError)])
def test_python_threads(threads, exception):
    with pytest.raises(exception):
        h.h_m_roth_primal_lp(G, 1, num_threads=threads)

def test_input_layouts_and_no_mutation():
    padded = np.zeros((2, 6))
    padded[:, ::2] = G
    readonly = G.copy()
    readonly.flags.writeable = False
    for matrix in (G.tolist(), np.asfortranarray(G), padded[:, ::2], G.astype(np.float32), readonly):
        before = np.asarray(matrix).copy()
        assert h.h_m_roth_primal_lp(matrix, np.int64(1)) == pytest.approx(2.)
        np.testing.assert_array_equal(matrix, before)

def test_profiles_and_zero_redundancy():
    profile = h.h_m_roth_primal_combinatorial(G)
    assert isinstance(profile, list)
    assert profile == pytest.approx([2.])
    assert h.h_m_roth_primal_combinatorial(np.eye(2)) == []
    assert h.h_m_roth_primal_combinatorial(np.eye(2), 0) == 1.
    assert h.h_m_roth_mds_combinatorial(np.eye(2), 0) == 0.
    assert h.h_m_roth_mds_combinatorial_parity(np.empty((0, 2)), 0) == 0.

def test_missing_native_does_not_fall_back():
    from analog_ecc_heights.cpp_backend import adapter
    with mock.patch.object(adapter, "load_native", side_effect=ImportError("missing test backend")):
        assert h.available_backends() == ["python"]
        assert h.h_m_roth_primal_lp(G, 1) == pytest.approx(2.)
        with pytest.raises(ImportError, match="missing test backend"):
            h.h_m_roth_primal_lp(G, 1, backend="cpp-glpk")

def test_default_call_never_loads_native():
    code = '''
import importlib.abc, json, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.endswith("solve_m_height_cpp") or fullname in tuple("analog_ecc_heights.cpp_backend." + x for x in ("_comb", "_glpk", "_highs"))):
            raise AssertionError("default call attempted native import")
sys.meta_path.insert(0, BlockNative())
import analog_ecc_heights as h
v = h.h_m_roth_primal_lp([[1, 0, 1], [0, 1, 1]], 1)
print(json.dumps([v, any(n == "solve_m_height_cpp" or n in tuple("analog_ecc_heights.cpp_backend." + x for x in ("_comb", "_glpk", "_highs")) for n in sys.modules)]))
'''
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                            text=True, env=os.environ.copy(), timeout=30)
    assert json.loads(result.stdout) == [2., False]


def test_native_parallel_defaults():
    from types import SimpleNamespace
    from analog_ecc_heights.cpp_backend import adapter
    native = SimpleNamespace(h_m_roth_primal_combinatorial=lambda G, m, **kw: float(kw["num_threads"]))
    with mock.patch.object(adapter, "load_native", return_value=native):
        assert h.h_m_roth_primal_combinatorial(G, 1, backend="cpp") == 16.
        assert h.h_m_roth_primal_combinatorial(G, 1, backend="cpp", num_threads=1) == 1.
        assert h.h_m_roth_primal_combinatorial(G, 1, backend="cpp", num_threads=2) == 2.


@pytest.mark.parametrize("missing", ["cpp", "cpp-glpk", "cpp-highs"])
def test_backend_availability_is_independent(missing):
    from analog_ecc_heights.cpp_backend import adapter
    def load(backend):
        if backend == missing:
            raise ImportError("missing library")
        return object()
    with mock.patch.object(adapter, "load_native", side_effect=load):
        assert h.available_backends() == [b for b in ("python", "cpp", "cpp-glpk", "cpp-highs") if b != missing]


def test_installed_native_missing_highs_does_not_use_glpk():
    from types import SimpleNamespace
    from analog_ecc_heights.cpp_backend import adapter
    glpk = mock.Mock(side_effect=AssertionError("GLPK must not be called"))
    with mock.patch.object(adapter, "load_native", return_value=SimpleNamespace(h_m_roth_primal_lp_glpk=glpk)):
        with pytest.raises(ImportError, match="cpp-highs"):
            h.h_m_roth_primal_lp(G, 1, backend="cpp-highs")
    glpk.assert_not_called()
