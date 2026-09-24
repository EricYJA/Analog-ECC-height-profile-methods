"""Benchmark Roth methods on random systematic integer codes.

Run from the repository root with the package installed::

    python evaluation/test_all_methods.py --output-dir evaluation/roth_results
    python evaluation/test_all_methods.py --output-dir evaluation/roth_results --resume

Defaults sweep 5 <= n <= 16, 2 <= k <= n-2, 2 <= m <= n-k,
with 50 accepted matrices per triple. A small run is, for example::

    python evaluation/test_all_methods.py --n-max 5 --samples 2 --output-dir /tmp/roth-small

summary.csv contains method/backend averages; details.csv contains individual
results. checkpoints/*.json store the actual matrices, RNG state, and each
completed call. Resume requires the same experiment options and installed
backends. --max-attempts is a per-matrix sampling budget for each invocation;
resuming an exhausted combination continues its random stream with a new budget.
Errors are recorded and retained on resume. Use a new output directory to retest
after changing implementations. Run only one process per output directory.

Agreement covers all applicable selected methods/backends and requires finite
heights. Pairwise comparisons use math.isclose (symmetric relative tolerance).
MDS methods apply only at m=n-k. Averages include successful finite calls only;
counts and statuses expose failures and sampling shortfalls. An empty agreement
cell means incomplete, unless an observed failure already establishes False.
Exit codes: 0 = complete and agreeing, 1 = incomplete/disagreement/method error,
2 = configuration/setup error, 130 = interrupted.
Timings include each public method's validation,
but exclude sampling, parity conversion, warmup, comparisons, and checkpoint I/O.
summary.csv is refreshed after each triple and on a graceful interruption;
details.csv is flushed after each completed matrix and rebuilt on resume.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

import analog_ecc_heights as heights
from analog_ecc_heights.py_backend._linalg import (
    generator_minimum_distance_exceeds_validated,
)


BACKENDS = ("python", "cpp", "cpp-glpk", "cpp-highs")
METHOD_NAMES = (
    "h_m_roth_primal_lp",
    "h_m_roth_dual_lp",
    "h_m_roth_primal_combinatorial",
    "h_m_roth_primal_combinatorial_pruning",
    "h_m_roth_dual_combinatorial_generator",
    "h_m_roth_dual_combinatorial_parity",
    "h_m_roth_mds_combinatorial",
    "h_m_roth_mds_combinatorial_parity",
)
SUMMARY_FIELDS = (
    "n", "k", "m", "method", "backend", "num_threads", "status",
    "requested_samples", "generated_samples", "completed_samples",
    "successful_calls", "failed_calls", "average_seconds", "sampling_attempts",
    "agreeing_samples", "disagreeing_samples", "all_methods_agree",
    "max_abs_difference", "rtol", "atol", "rank_tol",
)
DETAIL_FIELDS = (
    "n", "k", "m", "sample_index", "method", "backend", "height", "seconds",
    "status", "error", "all_methods_agree", "max_abs_difference", "checkpoint",
)


@dataclass(frozen=True)
class Method:
    name: str
    backend: str

    @property
    def key(self):
        return f"{self.name}:{self.backend}"

    def applies(self, n, k, m):
        return "_mds_" not in self.name or m == n - k

    def threads(self, requested):
        return requested if self.backend in ("cpp", "cpp-highs") else 1


def generator_to_parity_check(G):
    """Convert a systematic real generator G=[I_k | P] to H=[-P.T | I].

    This helper deliberately requires systematic input; it does not compute a
    nullspace for an arbitrary generator. H has full row rank and G @ H.T = 0.
    """
    raw = np.asarray(G)
    if np.iscomplexobj(raw):
        raise ValueError("G must be real.")
    G = np.asarray(raw, dtype=float)
    if G.ndim != 2 or not np.all(np.isfinite(G)):
        raise ValueError("G must be a finite two-dimensional matrix.")
    k, n = G.shape
    if not 0 < k <= n or not np.array_equal(G[:, :k], np.eye(k)):
        raise ValueError("G must be systematic with a leading identity block.")
    return np.column_stack((-G[:, k:].T, np.eye(n - k)))


def parameter_triples(args):
    for n in range(args.n_min, args.n_max + 1):
        for k in range(args.k_min, min(n - 2, args.k_max or n - 2) + 1):
            for m in range(args.m_min, min(n - k, args.m_max or n - k) + 1):
                yield n, k, m


def build_methods(backends):
    return [
        Method(name, backend)
        for name in METHOD_NAMES
        for backend in backends
        if backend == "python" or (
            backend == "cpp" if "combinatorial" in name
            else backend in ("cpp-glpk", "cpp-highs")
        )
    ]


def minimum_distance_exceeds(G, m, tol):
    # Every generator row is a codeword: cheaply reject many bad candidates
    # before the exhaustive rank checks, especially zero entries in MDS cases.
    if np.any(np.count_nonzero(G, axis=1) <= m):
        return False
    # Full row rank is guaranteed by the systematic identity block. The package
    # uses complete-pivot LU with tol times the parent matrix's entry scale.
    return generator_minimum_distance_exceeds_validated(G, m, tol)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def checkpoint_path(output, triple):
    n, k, m = triple
    return output / "checkpoints" / f"n{n}_k{k}_m{m}.json"


def load_state(path, triple, seed):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(np.random.SeedSequence([seed, *triple]))
    return dict(n=triple[0], k=triple[1], m=triple[2], attempts=0,
                sampling_exhausted=False, rng_state=rng.bit_generator.state,
                samples=[])


def generate_sample(state, args, path):
    """Rejection-sample uniformly from the integer ensemble, saving progress."""
    n, k, m = state["n"], state["k"], state["m"]
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    state["sampling_exhausted"] = False
    for attempt in range(1, args.max_attempts + 1):
        P = rng.integers(-5, 6, size=(k, n - k))
        G = np.column_stack((np.eye(k, dtype=int), P))
        state["attempts"] += 1
        state["rng_state"] = rng.bit_generator.state
        if minimum_distance_exceeds(G, m, args.rank_tol):
            state["samples"].append({"G": G.tolist(), "results": {}})
            write_json(path, state)
            return True
        if attempt % 1000 == 0:
            write_json(path, state)
            print(f"  ({n},{k},{m}): rejected {attempt}/{args.max_attempts} "
                  "candidates for the next matrix", flush=True)
    state["sampling_exhausted"] = True
    write_json(path, state)
    print(f"  ({n},{k},{m}): sampling budget exhausted; "
          f"only {len(state['samples'])}/{args.samples} matrices generated. "
          "Resume to try more candidates.", file=sys.stderr, flush=True)
    return False


def call_options(method, args):
    options = dict(backend=method.backend, num_threads=method.threads(args.num_threads))
    if "combinatorial" in method.name:
        options["tol"] = args.rank_tol
    return options


def warm_up(methods, args):
    G = np.array([[1., 0., 1., 1.], [0., 1., 1., 2.]])
    H = generator_to_parity_check(G)
    for method in methods:
        matrix = H if method.name.endswith("_parity") else G
        getattr(heights, method.name)(matrix, 2, **call_options(method, args))


def evaluate(method, G, H, m, args):
    matrix = H if method.name.endswith("_parity") else G
    function = getattr(heights, method.name)
    options = call_options(method, args)
    start = perf_counter()
    try:
        value = float(function(matrix, m, **options))
    except Exception as exc:
        return dict(height=None, seconds=perf_counter() - start, status="error",
                    error=f"{type(exc).__name__}: {exc}")
    elapsed = perf_counter() - start
    return dict(height=value if math.isfinite(value) else str(value), seconds=elapsed,
                status="ok" if math.isfinite(value) else "nonfinite",
                error="" if math.isfinite(value) else "Expected a finite height because d > m.")


def sample_agreement(sample, applicable, args):
    results = sample["results"]
    if any(method.key not in results for method in applicable):
        return None, None
    if any(results[method.key]["status"] != "ok" for method in applicable):
        return False, None
    values = [results[method.key]["height"] for method in applicable]
    matches = all(math.isclose(a, b, rel_tol=args.rtol, abs_tol=args.atol)
                  for a, b in combinations(values, 2))
    return matches, max(values) - min(values)


def detail_rows(state, methods, args, path):
    triple = state["n"], state["k"], state["m"]
    applicable = [method for method in methods if method.applies(*triple)]
    for index, sample in enumerate(state["samples"]):
        agree, difference = sample_agreement(sample, applicable, args)
        if agree is None:
            continue  # Partial calls are already durable in the checkpoint.
        for method in methods:
            result = sample["results"].get(method.key, {})
            yield dict(n=triple[0], k=triple[1], m=triple[2], sample_index=index,
                       method=method.name, backend=method.backend,
                       height=result.get("height"), seconds=result.get("seconds"),
                       status=result.get("status", "not_applicable"),
                       error=result.get("error", ""), all_methods_agree=agree,
                       max_abs_difference=difference,
                       checkpoint=str(path.relative_to(args.output_dir)))


def summary_rows(state, methods, args):
    triple = state["n"], state["k"], state["m"]
    applicable = [method for method in methods if method.applies(*triple)]
    comparisons = [sample_agreement(sample, applicable, args) for sample in state["samples"]]
    agreeing = sum(agree is True for agree, _ in comparisons)
    disagreeing = sum(agree is False for agree, _ in comparisons)
    completed = agreeing + disagreeing
    all_agree = False if disagreeing else (True if completed == args.samples else None)
    differences = [difference for _, difference in comparisons if difference is not None]
    for method in methods:
        results = [sample["results"][method.key] for sample in state["samples"]
                   if method.key in sample["results"]]
        successful = [result for result in results if result["status"] == "ok"]
        failed = len(results) - len(successful)
        if not method.applies(*triple):
            status = "not_applicable"
        elif state["sampling_exhausted"]:
            status = "sampling_exhausted"
        elif failed:
            status = "failed"
        else:
            status = "complete" if len(results) == args.samples else "incomplete"
        yield dict(n=triple[0], k=triple[1], m=triple[2], method=method.name,
                   backend=method.backend, num_threads=method.threads(args.num_threads),
                   status=status, requested_samples=args.samples,
                   generated_samples=len(state["samples"]), completed_samples=completed,
                   successful_calls=len(successful), failed_calls=failed,
                   average_seconds=(sum(r["seconds"] for r in successful) / len(successful)
                                    if successful else None),
                   sampling_attempts=state["attempts"], agreeing_samples=agreeing,
                   disagreeing_samples=disagreeing, all_methods_agree=all_agree,
                   max_abs_difference=max(differences) if differences else None,
                   rtol=args.rtol, atol=args.atol, rank_tol=args.rank_tol)


def write_summary(output, summaries):
    path = output / "summary.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for rows in summaries.values():
            writer.writerows(rows)
    temporary.replace(path)


def prepare_output(args, backends, available):
    config = {name: getattr(args, name) for name in (
        "n_min", "n_max", "k_min", "k_max", "m_min", "m_max", "samples",
        "seed", "rank_tol", "rtol", "atol", "num_threads",
    )}
    config.update(backends=backends, package_version=heights.__version__,
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  python_version=platform.python_version())
    manifest = dict(schema_version=1, config=config,
                    unavailable_backends=[b for b in BACKENDS if b not in available])
    path = args.output_dir / "manifest.json"
    if args.resume:
        if not path.exists():
            raise ValueError("--resume requires an existing manifest.json in --output-dir.")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("schema_version") != 1 or previous.get("config") != config:
            raise ValueError("Resume configuration differs from manifest.json; use the same "
                             "experiment options and environment, or a new output directory.")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Output directory is not empty; use --resume or a new directory.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)


def run(args):
    available = heights.available_backends()
    backends = [b for b in BACKENDS if b in (args.backends or available)]
    missing = set(backends) - set(available)
    if missing:
        raise ValueError(f"Requested backends are unavailable: {', '.join(sorted(missing))}")
    methods = build_methods(backends)
    triples = list(parameter_triples(args))
    if not triples:
        raise ValueError("The requested ranges contain no valid (n,k,m) triples.")
    prepare_output(args, backends, available)
    print(f"{len(triples)} triples, {args.samples} matrices each; backends: {', '.join(backends)}", flush=True)
    unavailable = [b for b in BACKENDS if b not in available]
    if unavailable:
        print(f"Unavailable backends (excluded): {', '.join(unavailable)}", flush=True)

    summaries = {}
    needs_work = False
    # Rebuild derived CSVs from authoritative checkpoints; a crash between
    # saving a result and writing CSV can neither lose nor duplicate results.
    with (args.output_dir / "details.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for triple in triples:
            path = checkpoint_path(args.output_dir, triple)
            state = load_state(path, triple, args.seed)
            summaries[triple] = list(summary_rows(state, methods, args))
            writer.writerows(detail_rows(state, methods, args, path))
            needs_work |= summaries[triple][0]["completed_samples"] < args.samples
        stream.flush()
        write_summary(args.output_dir, summaries)
        if needs_work:
            print("Warming up selected methods (untimed)...", flush=True)
            warm_up(methods, args)
        for triple in triples:
            path = checkpoint_path(args.output_dir, triple)
            state = load_state(path, triple, args.seed)
            applicable = [method for method in methods if method.applies(*triple)]
            try:
                for index in range(args.samples):
                    if index == len(state["samples"]) and not generate_sample(state, args, path):
                        break
                    sample = state["samples"][index]
                    if sample_agreement(sample, applicable, args)[0] is not None:
                        continue
                    G = np.asarray(sample["G"], dtype=float)
                    H = generator_to_parity_check(G)
                    print(f"({triple[0]},{triple[1]},{triple[2]}) matrix {index + 1}/{args.samples}", flush=True)
                    # Rotate method order to distribute order-related timing bias.
                    offset = index % len(applicable)
                    for method in applicable[offset:] + applicable[:offset]:
                        if method.key in sample["results"]:
                            continue
                        print(f"  {method.key}", flush=True)
                        sample["results"][method.key] = evaluate(method, G, H, triple[2], args)
                        write_json(path, state)
                    agree, difference = sample_agreement(sample, applicable, args)
                    print(f"  all methods agree: {agree}; max absolute difference: {difference}", flush=True)
                    # Only this newly completed sample has not yet been exported.
                    for row in detail_rows({**state, "samples": [sample]}, methods, args, path):
                        row["sample_index"] = index
                        writer.writerow(row)
                    stream.flush()
            finally:
                # Includes partial timings when Ctrl-C interrupts a long method.
                summaries[triple] = list(summary_rows(state, methods, args))
                write_summary(args.output_dir, summaries)

    incomplete = sum(rows[0]["completed_samples"] != args.samples for rows in summaries.values())
    disagreements = sum(rows[0]["disagreeing_samples"] for rows in summaries.values())
    print(f"Results: {args.output_dir / 'summary.csv'} and {args.output_dir / 'details.csv'}\n"
          f"Incomplete triples: {incomplete}; matrices with disagreement/errors: {disagreements}", flush=True)
    return 1 if incomplete or disagreements else 0


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-min", type=positive_integer, default=5)
    parser.add_argument("--n-max", type=positive_integer, default=16)
    parser.add_argument("--k-min", type=positive_integer, default=2)
    parser.add_argument("--k-max", type=positive_integer, default=None)
    parser.add_argument("--m-min", type=positive_integer, default=2)
    parser.add_argument("--m-max", type=positive_integer, default=None)
    parser.add_argument("--samples", type=positive_integer, default=50)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-attempts", type=positive_integer, default=10000,
                        help="candidate limit per new matrix per invocation (default: 10000)")
    parser.add_argument("--rank-tol", type=nonnegative_float, default=1e-10)
    parser.add_argument("--rtol", type=nonnegative_float, default=1e-8)
    parser.add_argument("--atol", type=nonnegative_float, default=1e-9)
    parser.add_argument("--num-threads", type=positive_integer, default=1,
                        help="workers for cpp/cpp-highs; python/cpp-glpk always use 1")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        help="default: every installed backend")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/roth_results"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (args.n_min < 5 or args.n_max > 16 or args.n_min > args.n_max
            or args.k_min < 2 or args.m_min < 2 or args.seed < 0
            or (args.k_max is not None and args.k_max < args.k_min)
            or (args.m_max is not None and args.m_max < args.m_min)):
        parser.error("require 5 <= n_min <= n_max <= 16, k_min,m_min >= 2, "
                     "maxima >= minima, and seed >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted. Completed calls are checkpointed; rerun with --resume.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
