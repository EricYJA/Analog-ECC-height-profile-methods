"""Lazy compatibility access to the original native function names."""

from analog_ecc_heights.cpp_backend.adapter import load_native as _load_native

_SYMBOLS = {
    "cpp": (
        "h_m_roth_primal_combinatorial", "h_m_roth_primal_combinatorial_pruning",
        "h_m_roth_dual_combinatorial_generator", "h_m_roth_dual_combinatorial_parity",
        "h_m_roth_mds_combinatorial", "h_m_roth_mds_combinatorial_parity",
    ),
}
for _solver in ("glpk", "highs"):
    _SYMBOLS[f"cpp-{_solver}"] = (
        f"h_m_jiang_original_lp_{_solver}",
        f"h_m_jiang_simplified_lp_{_solver}_early_quit",
        f"h_m_roth_primal_lp_{_solver}", f"h_m_roth_dual_lp_{_solver}",
    )


def __getattr__(name):
    if name == "__all__":
        return __dir__()
    for backend, symbols in _SYMBOLS.items():
        if name in symbols:
            try:
                return getattr(_load_native(backend), name)
            except ImportError as exc:
                raise AttributeError(f"{name} is unavailable: {exc}") from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = []
    for backend, symbols in _SYMBOLS.items():
        try:
            native = _load_native(backend)
        except ImportError:
            continue
        names.extend(name for name in symbols if hasattr(native, name))
    return sorted(names)
