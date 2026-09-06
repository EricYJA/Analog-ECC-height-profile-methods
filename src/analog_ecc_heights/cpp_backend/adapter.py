"""Adapt the stable public API to the original pybind11 function names."""

from importlib import import_module


_MODULES = {"cpp": "_comb", "cpp-glpk": "_glpk", "cpp-highs": "_highs"}
_REQUIREMENTS = {
    "cpp": "Eigen headers and OpenMP; build with -Cwheel.cmake=true",
    "cpp-glpk": "Eigen headers and system GLPK; build with -Cwheel.cmake=true "
                "-Ccmake.define.BUILD_CPP_COMB=OFF -Ccmake.define.USE_GLPK=ON",
    "cpp-highs": "Eigen headers, OpenMP, and system HiGHS; build with -Cwheel.cmake=true "
                 "-Ccmake.define.BUILD_CPP_COMB=OFF -Ccmake.define.USE_HIGHS=ON",
}


def load_native(backend):
    """Load only the extension for the requested backend."""
    try:
        return import_module(f".{_MODULES[backend]}", __package__)
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"Backend {backend!r} could not be loaded. Requires {_REQUIREMENTS[backend]}. "
            "Keep the native shared libraries available after installation. "
            f"Original error: {exc}"
        ) from exc


def _threads(backend, requested):
    if backend == "cpp-glpk":
        if requested not in (None, 1):
            raise ValueError("The cpp-glpk backend supports num_threads=None or 1 only.")
        return 1
    # OpenMP is required when building either parallel extension.
    return 16 if requested is None else requested


def call(method, matrix, m, *, backend, num_threads, **options):
    native = load_native(backend)
    if backend == "cpp":
        symbol = method
    else:
        solver = backend.removeprefix("cpp-")
        symbol = f"{method}_{solver}"
        if method == "h_m_jiang_simplified_lp":
            symbol += "_early_quit"
    try:
        function = getattr(native, symbol)
    except AttributeError as exc:
        raise ImportError(
            f"Backend {backend!r} is missing symbol {symbol!r}; rebuild analog-ecc-heights."
        ) from exc
    threads = _threads(backend, num_threads)
    if backend != "cpp-glpk":
        options["num_threads"] = threads
    return function(matrix, m, **options)
