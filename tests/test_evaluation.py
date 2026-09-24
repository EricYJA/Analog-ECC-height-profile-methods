"""Correctness and interruption checks for the standalone Roth benchmark."""

import csv
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "evaluation" / "test_all_methods.py"
if not SCRIPT.is_file():
    pytest.skip("Standalone evaluation scripts are not included in source distributions.",
                allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("roth_evaluation", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def small_args(tmp_path, *extra):
    return benchmark.parse_args([
        "--n-max", "5", "--k-max", "2", "--m-max", "2",
        "--samples", "2", "--backends", "python",
        "--output-dir", str(tmp_path / "results"), *extra,
    ])


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fake_result(value=3.0, seconds=0.25, status="ok"):
    return dict(height=value, seconds=seconds, status=status, error="")


def test_default_inclusive_sweep():
    args = benchmark.parse_args([])
    triples = list(benchmark.parameter_triples(args))
    expected = {(n, k, m) for n in range(5, 17)
                for k in range(2, n - 1) for m in range(2, n - k + 1)}
    assert set(triples) == expected
    assert len(triples) == 454
    assert len(triples) * args.samples == 22700


@pytest.mark.parametrize("k,r", [(2, 3), (3, 2), (8, 8), (14, 2), (2, 0)])
def test_parity_conversion(k, r):
    P = np.random.default_rng(57).integers(-5, 6, (k, r))
    G = np.column_stack((np.eye(k), P))
    original = G.copy()
    H = benchmark.generator_to_parity_check(G)
    assert H.shape == (r, k + r)
    np.testing.assert_array_equal(G @ H.T, np.zeros((k, r)))
    np.testing.assert_array_equal(H[:, k:], np.eye(r))
    np.testing.assert_array_equal(G, original)


@pytest.mark.parametrize("G", [
    [1, 2], [[1, 1, 2], [0, 1, 3]], [[1, 0, float("nan")]],
    [[1j, 1]], np.zeros((0, 2)), np.ones((3, 2)),
])
def test_parity_rejects_invalid_or_nonsystematic_input(G):
    with pytest.raises(ValueError):
        benchmark.generator_to_parity_check(G)


def test_distance_filter_checks_linear_combinations_not_just_rows():
    G = np.array([[1, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    assert np.all(np.count_nonzero(G, axis=1) > 2)
    # Subtract the rows: a weight-two codeword remains.
    assert not benchmark.minimum_distance_exceeds(G, 2, 1e-10)
    good = np.array([[1, 0, 1, 1, 1], [0, 1, 1, 2, 3]])
    assert benchmark.minimum_distance_exceeds(good, 3, 1e-10)


def test_generated_integer_matrices_have_required_distance_and_repeat(tmp_path):
    args = small_args(tmp_path)
    args.rank_tol = 1e-10
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        state = benchmark.load_state(path, (5, 2, 3), args.seed)
        for _ in range(3):
            assert benchmark.generate_sample(state, args, path)
            G = np.array(state["samples"][-1]["G"])
            assert np.issubdtype(G.dtype, np.integer)
            np.testing.assert_array_equal(G[:, :2], np.eye(2))
            assert np.all((-5 <= G[:, 2:]) & (G[:, 2:] <= 5))
            # Independent exact integer determinant test for d>3, k=2.
            for i, j in combinations(range(5), 2):
                assert G[0, i] * G[1, j] - G[0, j] * G[1, i] != 0
            state = benchmark.load_state(path, (5, 2, 3), args.seed)
    assert json.loads(paths[0].read_text()) == json.loads(paths[1].read_text())


def test_method_selection_and_mds_applicability():
    methods = benchmark.build_methods(benchmark.BACKENDS)
    assert len(methods) == 18
    assert {method.name for method in methods} == set(benchmark.METHOD_NAMES)
    assert sum(method.applies(5, 2, 2) for method in methods) == 14
    assert all(method.applies(5, 2, 3) for method in methods)
    for method in methods:
        assert method.threads(4) == (4 if method.backend in ("cpp", "cpp-highs") else 1)


def test_agreement_is_pairwise_and_requires_completed_finite_results(tmp_path):
    args = small_args(tmp_path)
    args.rtol, args.atol = 1e-8, 0
    methods = benchmark.build_methods(["python"])[:3]
    sample = {"results": {method.key: fake_result(value)
                         for method, value in zip(methods, [1., 1. - 0.75e-8, 1. + 0.75e-8])}}
    assert benchmark.sample_agreement(sample, methods, args)[0] is False
    sample["results"][methods[1].key]["height"] = 1.
    assert benchmark.sample_agreement(sample, methods, args)[0] is True
    sample["results"][methods[0].key] = fake_result("inf", status="nonfinite")
    assert benchmark.sample_agreement(sample, methods, args) == (False, None)
    del sample["results"][methods[-1].key]
    assert benchmark.sample_agreement(sample, methods, args) == (None, None)


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_evaluate_flags_nonfinite_results(tmp_path, monkeypatch, bad_value):
    args = small_args(tmp_path)
    method = benchmark.Method("h_m_roth_primal_lp", "python")
    monkeypatch.setattr(benchmark.heights, method.name, lambda *a, **kw: bad_value)
    result = benchmark.evaluate(method, np.eye(2), np.eye(2), 2, args)
    assert result["status"] == "nonfinite"
    assert result["seconds"] >= 0
    json.dumps(result, allow_nan=False)


def test_parity_method_receives_H_and_solver_errors_are_recorded(tmp_path, monkeypatch):
    args = small_args(tmp_path)
    method = benchmark.Method("h_m_roth_dual_combinatorial_parity", "python")
    G = np.array([[1, 0, 1, 1, 1], [0, 1, 1, 2, 3]])
    H = benchmark.generator_to_parity_check(G)

    def solver(matrix, m, **options):
        np.testing.assert_array_equal(matrix, H)
        assert m == 2 and options["tol"] == args.rank_tol
        raise RuntimeError("test solver failure")

    monkeypatch.setattr(benchmark.heights, method.name, solver)
    result = benchmark.evaluate(method, G, H, 2, args)
    assert result["status"] == "error"
    assert result["height"] is None
    assert "test solver failure" in result["error"]


def test_summary_averages_successful_calls_and_marks_incomplete(tmp_path):
    args = small_args(tmp_path, "--samples", "4")
    method = benchmark.Method("h_m_roth_primal_lp", "python")
    state = benchmark.load_state(tmp_path / "missing.json", (5, 2, 2), args.seed)
    state["samples"] = [
        {"results": {method.key: fake_result(seconds=1.)}},
        {"results": {method.key: fake_result(seconds=3.)}},
        {"results": {method.key: fake_result(None, seconds=7., status="error")}},
    ]
    row, = benchmark.summary_rows(state, [method], args)
    assert row["average_seconds"] == 2.
    assert row["successful_calls"] == 2 and row["failed_calls"] == 1
    assert row["all_methods_agree"] is False
    state["samples"].pop()
    row, = benchmark.summary_rows(state, [method], args)
    assert row["all_methods_agree"] is None
    assert row["status"] == "incomplete"


def test_interrupted_resume_preserves_timings_and_has_no_duplicate_csv_rows(tmp_path, monkeypatch):
    args = small_args(tmp_path)
    monkeypatch.setattr(benchmark, "warm_up", lambda *a: None)
    completed_calls = []

    def interrupt_after_two(method, *unused):
        if len(completed_calls) == 2:
            raise KeyboardInterrupt
        completed_calls.append(method.key)
        return fake_result(seconds=len(completed_calls))

    monkeypatch.setattr(benchmark, "evaluate", interrupt_after_two)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(args)
    path = benchmark.checkpoint_path(args.output_dir, (5, 2, 2))
    before = json.loads(path.read_text())
    assert len(before["samples"][0]["results"]) == 2
    assert all(row["all_methods_agree"] == "" for row in read_csv(args.output_dir / "summary.csv"))

    resumed_calls = []

    def finish(method, *unused):
        resumed_calls.append(method.key)
        return fake_result()

    monkeypatch.setattr(benchmark, "evaluate", finish)
    args.resume = True
    assert benchmark.run(args) == 0
    assert len(resumed_calls) == 10  # Six applicable methods per sample, two saved.
    after = json.loads(path.read_text())
    for key, result in before["samples"][0]["results"].items():
        assert after["samples"][0]["results"][key] == result
    rows = read_csv(args.output_dir / "details.csv")
    assert len(rows) == 16  # Includes two inapplicable MDS rows per matrix.
    assert len({(row["sample_index"], row["method"], row["backend"]) for row in rows}) == 16
    assert all(row["all_methods_agree"] == "True" for row in rows)
    snapshot = (args.output_dir / "details.csv").read_bytes()
    resumed_calls.clear()
    assert benchmark.run(args) == 0
    assert not resumed_calls
    assert (args.output_dir / "details.csv").read_bytes() == snapshot

    args.seed += 1
    with pytest.raises(ValueError, match="configuration differs"):
        benchmark.run(args)
    assert (args.output_dir / "details.csv").read_bytes() == snapshot


def test_sampling_exhaustion_is_not_success_and_resume_continues_rng(tmp_path, monkeypatch):
    args = small_args(tmp_path, "--max-attempts", "2")
    original_filter = benchmark.minimum_distance_exceeds
    monkeypatch.setattr(benchmark, "warm_up", lambda *a: None)
    monkeypatch.setattr(benchmark, "minimum_distance_exceeds", lambda *a: False)
    assert benchmark.run(args) == 1
    rows = read_csv(args.output_dir / "summary.csv")
    assert rows[0]["status"] == "sampling_exhausted"
    assert rows[0]["all_methods_agree"] == ""
    path = benchmark.checkpoint_path(args.output_dir, (5, 2, 2))
    before = json.loads(path.read_text())
    assert before["attempts"] == 2 and before["samples"] == []

    monkeypatch.setattr(benchmark, "minimum_distance_exceeds", original_filter)
    monkeypatch.setattr(benchmark, "evaluate", lambda *a: fake_result())
    args.resume, args.max_attempts = True, 100
    assert benchmark.run(args) == 0
    after = json.loads(path.read_text())
    assert after["attempts"] >= 4
    assert after["rng_state"] != before["rng_state"]
    assert len(after["samples"]) == 2


def test_disagreement_sets_exit_status_and_csv_flag(tmp_path, monkeypatch):
    args = small_args(tmp_path)
    monkeypatch.setattr(benchmark, "warm_up", lambda *a: None)
    monkeypatch.setattr(benchmark, "evaluate", lambda method, *a:
                        fake_result(4. if method.name.endswith("primal_lp") else 3.))
    assert benchmark.run(args) == 1
    rows = read_csv(args.output_dir / "summary.csv")
    assert all(row["all_methods_agree"] == "False" for row in rows)
    assert all(row["disagreeing_samples"] == "2" for row in rows)


def test_unavailable_requested_backend_fails_before_writing(tmp_path, monkeypatch):
    args = small_args(tmp_path)
    args.backends = ["cpp-glpk"]
    monkeypatch.setattr(benchmark.heights, "available_backends", lambda: ["python"])
    with pytest.raises(ValueError, match="unavailable"):
        benchmark.run(args)
    assert not args.output_dir.exists()
