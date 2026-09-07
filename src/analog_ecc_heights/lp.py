"""Linear-programming methods for analog error-correcting code heights."""

import numpy as np

from . import _validation as validate
from ._dispatch import run


def _solve(method, G, m, backend, num_threads, **options):
    G = validate.matrix(G)
    jiang = "jiang" in method
    m = validate.height_index(m, G.shape[1], jiang=jiang)
    if method == "h_m_jiang_original_lp" and np.any(np.all(G == 0, axis=0)):
        raise ValueError("Jiang original LP assumes no zero column in G.")
    if "early_quit_threshold" in options:
        options["early_quit_threshold"] = validate.threshold(options["early_quit_threshold"])
    return float(run(method, G, m, backend=backend, num_threads=num_threads, **options))


def h_m_jiang_original_lp(G, m, *, backend="python", num_threads=None):
    """Compute Jiang's original LP formulation.

    G must be real, finite and have no zero columns. Requires
    1 <= m < n. No early-quit threshold is supported.
    Backends: python (default), cpp-glpk, cpp-highs.
    """
    return _solve("h_m_jiang_original_lp", G, m, backend, num_threads)


def h_m_jiang_simplified_lp(
    G, m, *, backend="python", early_quit_threshold=float("inf"), num_threads=None
):
    """Compute Jiang's simplified LP, returning min(height, threshold).

    Requires 1 <= m < n. An infinite threshold computes the
    complete height. Backends: python (default), cpp-glpk, cpp-highs.
    """
    return _solve("h_m_jiang_simplified_lp", G, m, backend, num_threads,
                  early_quit_threshold=early_quit_threshold)


def h_m_roth_primal_lp(
    G, m, *, backend="python", early_quit_threshold=float("inf"), num_threads=None
):
    """Compute Roth's primal LP, returning min(height, threshold).

    Requires 0 <= m < n. At m=0, returns min(1, threshold).
    Backends: python (default), cpp-glpk, cpp-highs.
    """
    return _solve("h_m_roth_primal_lp", G, m, backend, num_threads,
                  early_quit_threshold=early_quit_threshold)


def h_m_roth_dual_lp(
    G, m, *, backend="python", early_quit_threshold=float("inf"), num_threads=None
):
    """Compute Roth's dual LP, returning min(height, threshold).

    Requires 0 <= m < n. At m=0, returns min(1, threshold).
    Backends: python (default), cpp-glpk, cpp-highs.
    """
    return _solve("h_m_roth_dual_lp", G, m, backend, num_threads,
                  early_quit_threshold=early_quit_threshold)
