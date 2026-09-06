"""Compare available implementations on a small known-answer matrix."""

import math
from time import perf_counter

import numpy as np

from analog_ecc_heights import (
    available_backends,
    h_m_roth_primal_combinatorial,
    h_m_roth_primal_lp,
)


def main():
    G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    available = set(available_backends())
    expected = h_m_roth_primal_lp(G, 1)
    print("Small example timings include solver setup and are not a benchmark.")
    print(f"{'Method':<16} {'Backend':<12} {'Height':>10} {'Seconds':>10}")
    for family, function, backends in (
        ("LP", h_m_roth_primal_lp, ("python", "cpp-glpk", "cpp-highs")),
        ("Combinatorial", h_m_roth_primal_combinatorial, ("python", "cpp")),
    ):
        for backend in backends:
            if backend not in available:
                print(f"{family:<16} {backend:<12} skipped (not installed)")
                continue
            start = perf_counter()
            height = function(G, 1, backend=backend, num_threads=1)
            seconds = perf_counter() - start
            if not math.isclose(height, expected, rel_tol=1e-7, abs_tol=1e-9):
                raise RuntimeError(f"{family}/{backend} returned {height}, expected {expected}")
            print(f"{family:<16} {backend:<12} {height:>10.6g} {seconds:>10.6f}")


if __name__ == "__main__":
    main()
