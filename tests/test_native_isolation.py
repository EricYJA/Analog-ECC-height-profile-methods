"""Check that native solver dependencies cannot leak between extensions."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
from analog_ecc_heights.cpp_backend.adapter import load_native


@pytest.mark.parametrize("backend", ("cpp", "cpp-glpk", "cpp-highs"))
def test_requested_backend_does_not_import_other_extensions(backend):
    try:
        load_native(backend)
    except ImportError:
        pytest.skip(f"{backend} not installed")
    script = r"""
import importlib.abc
import sys
selected = sys.argv[1]
modules = {"cpp": "_comb", "cpp-glpk": "_glpk", "cpp-highs": "_highs"}
forbidden = {"analog_ecc_heights.cpp_backend." + module
             for backend, module in modules.items() if backend != selected}
class BlockOthers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in forbidden:
            raise AssertionError(f"Attempted unrelated extension import: {fullname}")
sys.meta_path.insert(0, BlockOthers())
import analog_ecc_heights as h
import solve_m_height_cpp as legacy
assert not any("analog_ecc_heights.cpp_backend." + name in sys.modules for name in modules.values())
fn = h.h_m_roth_primal_combinatorial if selected == "cpp" else h.h_m_roth_primal_lp
assert abs(fn([[1., 1., 1.]], 1, backend=selected) - 1.) < 1e-9
symbol = "h_m_roth_primal_combinatorial" if selected == "cpp" else "h_m_roth_primal_lp_" + selected[4:]
args = ([[1., 1., 1.]], 1) if selected == "cpp" else ([[1., 1., 1.]], 1, float("inf"))
assert abs(getattr(legacy, symbol)(*args) - 1.) < 1e-9
assert forbidden.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, backend], check=True,
                   capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("readelf"),
                    reason="ELF linkage inspection requires Linux readelf")
@pytest.mark.parametrize("backend", ("cpp", "cpp-glpk", "cpp-highs"))
def test_native_link_dependencies(backend):
    try:
        native = load_native(backend)
    except ImportError:
        pytest.skip(f"{backend} not installed")
    result = subprocess.run(["readelf", "-d", native.__file__], check=True,
                            capture_output=True, text=True)
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", result.stdout)
    has = lambda text: any(text in library.lower() for library in needed)
    assert has("glpk") == (backend == "cpp-glpk"), needed
    assert has("highs") == (backend == "cpp-highs"), needed
    assert (has("libgomp") or has("libomp") or has("libiomp")) == (backend != "cpp-glpk"), needed


@pytest.mark.skipif(os.environ.get("ANALOG_ECC_TEST_CMAKE") != "1",
                    reason="Set ANALOG_ECC_TEST_CMAKE=1 with native build dependencies installed")
@pytest.mark.parametrize("backend", ("cpp", "cpp-glpk", "cpp-highs"))
def test_openmp_build_requirement(backend, tmp_path):
    import pybind11
    root = Path(__file__).resolve().parents[1]
    command = ["cmake", "-S", str(root), "-B", str(tmp_path / "build"),
               f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
               "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE",
               f"-DBUILD_CPP_COMB={'ON' if backend == 'cpp' else 'OFF'}",
               f"-DUSE_GLPK={'ON' if backend == 'cpp-glpk' else 'OFF'}",
               f"-DUSE_HIGHS={'ON' if backend == 'cpp-highs' else 'OFF'}"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    if backend == "cpp-glpk":
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
        assert "OpenMP" in output and "REQUIRED" in output, output
