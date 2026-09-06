"""Public-API integration checks across installed execution backends.

Native cases skip in the Python-only wheel and run when the same distribution
is built with its optional compiled extension.
"""

from importlib import import_module

import numpy as np
import pytest

import analog_ecc_heights as heights


LP_METHODS = (
    heights.h_m_jiang_original_lp,
    heights.h_m_jiang_simplified_lp,
    heights.h_m_roth_primal_lp,
    heights.h_m_roth_dual_lp,
)
CAPPED_METHODS = LP_METHODS[1:]
GENERATOR_METHODS = (
    heights.h_m_roth_primal_combinatorial,
    heights.h_m_roth_primal_combinatorial_pruning,
    heights.h_m_roth_dual_combinatorial_generator,
)
LP_BACKENDS = ("python", "cpp-glpk", "cpp-highs")
COMB_BACKENDS = ("python", "cpp")


def require_backend(backend):
    if backend not in heights.available_backends():
        pytest.skip(f"Optional backend {backend!r} is not installed.")


def systematic_pair(P):
    k, r = P.shape
    return np.column_stack((np.eye(k), P)), np.column_stack((-P.T, np.eye(r)))


def assert_height(value, expected):
    assert isinstance(value, float)
    np.testing.assert_allclose(value, expected, rtol=1e-8, atol=1e-9)


@pytest.fixture(params=[
    np.array([[1.], [1.]]),
    np.eye(2),
    np.random.default_rng(714).uniform(-2., 2., (3, 2)),
], ids=["single-parity", "distance-two", "random-rank-three"])
def code_pair(request):
    return systematic_pair(request.param)


def test_available_backend_names_match_native_features():
    available = heights.available_backends()
    assert isinstance(available, list)
    assert available[0] == "python"
    assert len(available) == len(set(available))
    assert set(available) <= {"python", "cpp", "cpp-glpk", "cpp-highs"}
    if len(available) == 1:
        return
    native = import_module("analog_ecc_heights.cpp_backend.solve_m_height_cpp")
    for backend, symbol in (("cpp", "h_m_roth_primal_combinatorial"),
                            ("cpp-glpk", "h_m_roth_primal_lp_glpk"),
                            ("cpp-highs", "h_m_roth_primal_lp_highs")):
        assert (backend in available) == hasattr(native, symbol)


@pytest.mark.parametrize("backend", LP_BACKENDS)
def test_all_public_lp_formulations(backend, code_pair):
    require_backend(backend)
    G, _ = code_pair
    for m in range(1, G.shape[1]):
        expected = heights.h_m_roth_primal_lp(G, m)
        for method in LP_METHODS:
            assert_height(method(G, m, backend=backend, num_threads=1), expected)
    for method in LP_METHODS[2:]:
        assert_height(method(G, 0, backend=backend, num_threads=1), 1.)


@pytest.mark.parametrize("backend", COMB_BACKENDS)
def test_public_combinatorial_and_profile_agreement(backend, code_pair):
    require_backend(backend)
    G, H = code_pair
    profile_reference = []
    for m in range(G.shape[1]):
        expected = heights.h_m_roth_primal_lp(G, m)
        for method in GENERATOR_METHODS:
            assert_height(method(G, m, backend=backend, num_threads=1), expected)
        assert_height(heights.h_m_roth_dual_combinatorial_parity(
            H, m, backend=backend, num_threads=1), expected)
        if 1 <= m <= H.shape[0]:
            profile_reference.append(expected)
    profile = heights.h_m_roth_primal_combinatorial(G, backend=backend, num_threads=1)
    assert isinstance(profile, list)
    assert all(isinstance(value, float) for value in profile)
    np.testing.assert_allclose(profile, profile_reference, rtol=1e-8, atol=1e-9)
    assert profile == heights.h_m_roth_primal_combinatorial(
        G, None, backend=backend, num_threads=1)


@pytest.mark.parametrize("backend", LP_BACKENDS)
def test_caps_on_finite_and_infinite_heights(backend):
    require_backend(backend)
    G, _ = systematic_pair(np.array([[1.], [1.]]))
    # h_1 = 2 and h_2 = infinity for this code.
    for m, height in ((1, 2.), (2, float("inf"))):
        for threshold in (0., 0.5, 1.5, 3., float("inf")):
            for method in CAPPED_METHODS:
                assert_height(method(G, m, backend=backend, num_threads=1,
                                     early_quit_threshold=threshold),
                              min(height, threshold))
    for method in LP_METHODS[2:]:
        assert_height(method(G, 0, backend=backend, num_threads=1,
                             early_quit_threshold=0.5), 0.5)


@pytest.mark.parametrize("backend", LP_BACKENDS)
def test_rank_three_unbounded_case(backend):
    require_backend(backend)
    # This case exposed a SciPy presolve infeasible/unbounded ambiguity.
    G, _ = systematic_pair(np.random.default_rng(714).uniform(-2., 2., (3, 2)))
    for method in LP_METHODS:
        assert_height(method(G, 4, backend=backend, num_threads=1), float("inf"))


@pytest.mark.parametrize("backend", COMB_BACKENDS)
def test_dual_minimum_stays_per_target(backend):
    require_backend(backend)
    G, H = systematic_pair(np.array([[1., 3., 3., -2.], [-2., 1., 3., -2.]]))
    assert_height(heights.h_m_roth_dual_combinatorial_generator(
        G, 2, backend=backend, num_threads=1), 3.)
    assert_height(heights.h_m_roth_dual_combinatorial_parity(
        H, 2, backend=backend, num_threads=1), 3.)


@pytest.mark.parametrize("backend", COMB_BACKENDS)
def test_mds_specializations_and_zero_redundancy(backend):
    require_backend(backend)
    G, H = systematic_pair(np.array([[1., 1.], [1., 2.]]))
    expected = heights.h_m_roth_primal_lp(G, 2)
    for method, matrix in ((heights.h_m_roth_mds_combinatorial, G),
                           (heights.h_m_roth_mds_combinatorial_parity, H)):
        assert_height(method(matrix, 2, backend=backend, num_threads=1), expected)
    for method, matrix in ((heights.h_m_roth_mds_combinatorial, np.eye(3)),
                           (heights.h_m_roth_mds_combinatorial_parity, np.empty((0, 3)))):
        assert_height(method(matrix, 0, backend=backend, num_threads=1), 0.)
    bad_G, bad_H = systematic_pair(np.eye(2))
    for method, matrix in ((heights.h_m_roth_mds_combinatorial, bad_G),
                           (heights.h_m_roth_mds_combinatorial_parity, bad_H)):
        with pytest.raises(ValueError, match="MDS"):
            method(matrix, 2, backend=backend, num_threads=1)


@pytest.mark.parametrize("backend", COMB_BACKENDS)
def test_near_rank_parent_scale_is_preserved(backend):
    require_backend(backend)
    G = np.array([[1., 1e-12]])
    for tol, expected in ((1e-10, float("inf")), (1e-14, 1e12)):
        for method in GENERATOR_METHODS:
            assert_height(method(G, 1, backend=backend, tol=tol, num_threads=1), expected)
        np.testing.assert_allclose(heights.h_m_roth_primal_combinatorial(
            G, backend=backend, tol=tol, num_threads=1), [expected])


@pytest.mark.parametrize("backend", ("python", "cpp", "cpp-glpk", "cpp-highs"))
def test_array_layouts_and_real_input_conversion(backend):
    require_backend(backend)
    G, H = systematic_pair(np.array([[1., 1.], [1., 2.]]))
    if backend in COMB_BACKENDS:
        methods = [(method, G) for method in GENERATOR_METHODS]
        methods.append((heights.h_m_roth_dual_combinatorial_parity, H))
    else:
        methods = [(method, G) for method in LP_METHODS]
    if backend == "python":
        methods += [(method, G) for method in LP_METHODS]
    for method, matrix in methods:
        padded = np.empty((matrix.shape[0], 2 * matrix.shape[1]))
        padded[:, ::2] = matrix
        variants = (matrix.tolist(), matrix.astype(np.int64), matrix.astype(np.float32),
                    np.asfortranarray(matrix), padded[:, ::2], matrix[:, ::-1])
        expected = method(matrix, 2)
        for variant in variants:
            assert_height(method(variant, 2, backend=backend, num_threads=1), expected)


@pytest.mark.parametrize("backend", ("cpp", "cpp-highs"))
def test_parallel_requests_match_build_capabilities(backend):
    require_backend(backend)
    info = import_module("analog_ecc_heights.cpp_backend._build_info")
    assert isinstance(info.HAS_OPENMP, bool)
    G, H = systematic_pair(np.array([[1., 1.], [1., 2.]]))
    if backend == "cpp":
        methods = [(method, G) for method in GENERATOR_METHODS]
        methods += [(heights.h_m_roth_dual_combinatorial_parity, H),
                    (heights.h_m_roth_mds_combinatorial, G),
                    (heights.h_m_roth_mds_combinatorial_parity, H)]
    else:
        methods = [(method, G) for method in LP_METHODS]
    for method, matrix in methods:
        expected = method(matrix, 2, backend=backend, num_threads=1)
        assert_height(method(matrix, 2, backend=backend), expected)
        if info.HAS_OPENMP:
            assert_height(method(matrix, 2, backend=backend, num_threads=2), expected)
        else:
            with pytest.raises(ValueError, match="OpenMP"):
                method(matrix, 2, backend=backend, num_threads=2)
    if backend == "cpp" and info.HAS_OPENMP:
        np.testing.assert_allclose(heights.h_m_roth_primal_combinatorial(
            G, backend=backend, num_threads=2), heights.h_m_roth_primal_combinatorial(G))


def test_glpk_rejects_parallel_workers():
    require_backend("cpp-glpk")
    G, _ = systematic_pair(np.array([[1.], [1.]]))
    with pytest.raises(ValueError, match="num_threads"):
        heights.h_m_roth_primal_lp(G, 1, backend="cpp-glpk", num_threads=2)
