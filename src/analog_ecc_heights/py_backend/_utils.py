"""Small, deterministic enumeration helpers shared by Python methods."""

from itertools import combinations, product

import numpy as np


def complement_indices(n, subset):
    """Return the sorted complement of a coordinate subset."""
    excluded = set(subset)
    return [i for i in range(n) if i not in excluded]


def subsets(pool, size):
    """Yield lexicographic subsets; impossible sizes have no subsets."""
    if isinstance(pool, int):
        pool = range(pool)
    if size < 0 or size > len(pool):
        return iter(())
    return combinations(pool, size)


def sign_vectors(k):
    """Enumerate signs once modulo global sign symmetry, as native code does."""
    if k == 0:
        yield np.empty(0, dtype=float)
    elif k > 0:
        for tail in product((-1.0, 1.0), repeat=k - 1):
            yield np.asarray((1.0, *tail))
