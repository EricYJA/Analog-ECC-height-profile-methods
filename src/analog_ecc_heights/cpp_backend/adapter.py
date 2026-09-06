"""Adapt the stable public API to the original pybind11 function names."""

from importlib import import_module


def load_native():
    try:
        return import_module(".solve_m_height_cpp", __package__)
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The C++ extension could not be loaded. Install Eigen headers and GLPK, "
            "then build analog-ecc-heights from source with -Cwheel.cmake=true. "
            "For cpp-highs also install the HiGHS development library and add "
            "-Ccmake.define.USE_HIGHS=ON. If already built, check that the shared "
            f"libraries are available. Original error: {exc}"
        ) from exc


def _threads(backend, requested):
    if backend == "cpp-glpk":
        if requested not in (None, 1):
            raise ValueError("The cpp-glpk backend supports num_threads=None or 1 only.")
        return 1
    try:
        info = import_module("._build_info", __package__)
        parallel = info.HAS_OPENMP
    except ImportError:
        parallel = False
    if requested is None:
        return 16 if parallel else 1
    if requested > 1 and not parallel:
        raise ValueError("This C++ extension was built without OpenMP; use num_threads=1.")
    return requested


def call(method, matrix, m, *, backend, num_threads, **options):
    native = load_native()
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
            f"Backend {backend!r} is not present in this build. For cpp-highs, "
            "install the HiGHS development library and rebuild from source with "
            "-Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON."
        ) from exc
    threads = _threads(backend, num_threads)
    if backend != "cpp-glpk":
        options["num_threads"] = threads
    return function(matrix, m, **options)
