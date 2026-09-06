"""Explicit backend selection; native imports happen only on request."""

from importlib import import_module

from ._validation import thread_count


_METHOD_MODULES = {
    "h_m_jiang_original_lp": "jiang_lp",
    "h_m_jiang_simplified_lp": "jiang_lp",
    "h_m_roth_primal_lp": "roth_lp",
    "h_m_roth_dual_lp": "roth_lp",
    "h_m_roth_primal_combinatorial": "roth_comb",
    "h_m_roth_primal_combinatorial_pruning": "roth_comb",
    "h_m_roth_dual_combinatorial_generator": "roth_comb",
    "h_m_roth_dual_combinatorial_parity": "roth_comb",
    "h_m_roth_mds_combinatorial": "roth_mds_comb",
    "h_m_roth_mds_combinatorial_parity": "roth_mds_comb",
}


def run(method, matrix, m, *, backend, num_threads, **options):
    family = "combinatorial" if "combinatorial" in method else "lp"
    accepted = ("python", "cpp") if family == "combinatorial" else (
        "python", "cpp-glpk", "cpp-highs"
    )
    if not isinstance(backend, str) or backend not in accepted:
        raise ValueError(f"Unsupported backend {backend!r} for {family}; choose from {accepted}.")
    threads = thread_count(num_threads)
    if backend == "python":
        if threads not in (None, 1):
            raise ValueError("The python backend supports num_threads=None or 1 only.")
        module = import_module(f".py_backend.{_METHOD_MODULES[method]}", __package__)
        return getattr(module, method)(matrix, m, **options)
    from .cpp_backend.adapter import call

    return call(method, matrix, m, backend=backend, num_threads=threads, **options)


def available_backends():
    """Return usable backend names; this explicit check may load native code.

    ``python`` supports both method families, ``cpp`` supports combinatorial
    methods, and ``cpp-glpk``/``cpp-highs`` support LP methods.
    """
    from .cpp_backend.adapter import load_native

    result = ["python"]
    for backend in ("cpp", "cpp-glpk", "cpp-highs"):
        try:
            load_native(backend)
        except ImportError:
            continue
        result.append(backend)
    return result
