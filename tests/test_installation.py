"""Check release artifacts from an external working directory.

Set ANALOG_ECC_TEST_WHEEL to a built wheel and optionally ANALOG_ECC_TEST_SDIST
to its source archive. These checks do not build or download dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile

import pytest


@pytest.fixture(scope="module")
def built_wheel():
    configured = os.environ.get("ANALOG_ECC_TEST_WHEEL")
    if not configured:
        pytest.skip("Set ANALOG_ECC_TEST_WHEEL to validate a release wheel")
    wheel = Path(configured).resolve()
    assert wheel.is_file(), f"Wheel does not exist: {wheel}"
    assert wheel.suffix == ".whl"
    return wheel


@pytest.mark.installation
def test_wheel_contains_package_and_legacy_shim(built_wheel):
    with zipfile.ZipFile(built_wheel) as archive:
        names = archive.namelist()
        assert "analog_ecc_heights/__init__.py" in names
        assert "solve_m_height_cpp.py" in names
        assert not any("/cpp_backend/src/" in name for name in names)
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        assert "Name: analog-ecc-heights\n" in metadata
        assert "Version: 0.1.1\n" in metadata
        assert "Requires-Python: >=3.10" in metadata
        native = any(
            name.startswith(tuple(f"analog_ecc_heights/cpp_backend/_{part}." for part in ("comb", "glpk", "highs")))
            and name.endswith((".so", ".pyd"))
            for name in names
        )
        if os.environ.get("ANALOG_ECC_EXPECT_NATIVE") is not None:
            assert native, "The requested native build produced a wheel without its extension"
        wheel_metadata_name = next(name for name in names if name.endswith(".dist-info/WHEEL"))
        wheel_metadata = archive.read(wheel_metadata_name).decode()
        if native:
            assert "Root-Is-Purelib: false" in wheel_metadata
            assert "Tag: py3-none-any" not in wheel_metadata
            assert "analog_ecc_heights/cpp_backend/_build_info.py" in names
        else:
            assert "Root-Is-Purelib: true" in wheel_metadata
            assert "Tag: py3-none-any" in wheel_metadata
            assert not any(name.endswith((".so", ".pyd", ".dll", ".dylib")) for name in names)


@pytest.mark.installation
def test_installed_wheel_works_outside_checkout(built_wheel, tmp_path):
    # Reuse only third-party dependencies, avoiding downloads. The wheel itself
    # is installed into a fresh environment and its location is verified below.
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [
            str(executable), "-I", "-m", "pip", "--isolated", "install",
            "--no-index", "--no-deps", "--ignore-installed", str(built_wheel),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    with zipfile.ZipFile(built_wheel) as archive:
        native = any(
            name.startswith(tuple(f"analog_ecc_heights/cpp_backend/_{part}." for part in ("comb", "glpk", "highs")))
            and name.endswith((".so", ".pyd"))
            for name in archive.namelist()
        )
    script = r"""
import importlib.abc
import json
from pathlib import Path
import sys

class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "solve_m_height_cpp" or fullname in tuple("analog_ecc_heights.cpp_backend." + x for x in ("_comb", "_glpk", "_highs")):
            raise AssertionError("Default API attempted to load the native extension")

blocker = BlockNative()
sys.meta_path.insert(0, blocker)
import analog_ecc_heights as heights

assert Path(heights.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
G = [[1.0, 1.0, 1.0]]
assert abs(heights.h_m_roth_primal_lp(G, 1) - 1.0) < 1e-9
assert abs(heights.h_m_roth_primal_combinatorial(G, 1) - 1.0) < 1e-9
sys.meta_path.remove(blocker)

available = heights.available_backends()
import solve_m_height_cpp as legacy
for backend, symbol in (("cpp", "h_m_roth_primal_combinatorial"),
                        ("cpp-glpk", "h_m_roth_primal_lp_glpk"),
                        ("cpp-highs", "h_m_roth_primal_lp_highs")):
    fn = heights.h_m_roth_primal_combinatorial if backend == "cpp" else heights.h_m_roth_primal_lp
    if backend in available:
        assert abs(fn(G, 1, backend=backend) - 1.0) < 1e-9
        args = (G, 1) if backend == "cpp" else (G, 1, float("inf"))
        assert abs(getattr(legacy, symbol)(*args) - 1.0) < 1e-9
    else:
        try:
            fn(G, 1, backend=backend)
        except ImportError:
            pass
        else:
            raise AssertionError(f"Unexpected backend {backend}")
        assert not hasattr(legacy, symbol)
if sys.argv[2] == "native":
    from analog_ecc_heights.cpp_backend import _build_info
    assert _build_info.HAS_COMB == ("cpp" in available)
    assert _build_info.HAS_GLPK == ("cpp-glpk" in available)
    assert _build_info.HAS_HIGHS == ("cpp-highs" in available)
    assert _build_info.HAS_OPENMP == (_build_info.HAS_COMB or _build_info.HAS_HIGHS)
else:
    assert available == ["python"]
print(json.dumps(available))
"""
    result = subprocess.run(
        [str(executable), "-I", "-c", script, str(environment), "native" if native else "python"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    capabilities = json.loads(result.stdout.strip().splitlines()[-1])
    expected = os.environ.get("ANALOG_ECC_EXPECT_NATIVE")
    if expected is not None:
        names = {"comb": "cpp", "glpk": "cpp-glpk", "highs": "cpp-highs"}
        assert capabilities == ["python"] + [names[x] for x in ("comb", "glpk", "highs") if x in expected.split(",")]



@pytest.mark.installation
def test_sdist_includes_native_build_sources():
    configured = os.environ.get("ANALOG_ECC_TEST_SDIST")
    if not configured:
        pytest.skip("Set ANALOG_ECC_TEST_SDIST to validate a source archive")
    with tarfile.open(Path(configured).resolve(), "r:gz") as archive:
        names = {name.partition("/")[2] for name in archive.getnames()}
    assert {
        "pyproject.toml",
        "CMakeLists.txt",
        "cmake/build_info.py.in",
        "src/solve_m_height_cpp.py",
        *[f"src/analog_ecc_heights/cpp_backend/src/py_binding_{part}.cc" for part in ("comb", "glpk", "highs")],
        "src/analog_ecc_heights/cpp_backend/src/lp_common.hh",
        *[f"src/analog_ecc_heights/cpp_backend/src/lp_workspace_{part}.hh" for part in ("glpk", "highs")],
        "src/analog_ecc_heights/cpp_backend/src/methods.hh",
        "src/analog_ecc_heights/cpp_backend/src/utils.hh",
        *[f"src/analog_ecc_heights/cpp_backend/src/methods_{family}_lp_{part}.cc" for family in ("jiang", "roth") for part in ("glpk", "highs")],
        "src/analog_ecc_heights/cpp_backend/src/methods_roth_comb.cc",
        "src/analog_ecc_heights/cpp_backend/src/methods_roth_mds_comb.cc",
        "tests/native/test_main.cc",
    } <= names
    assert not any(name.endswith((".so", ".pyd", ".pyc")) for name in names)
    allowed_root_files = {"pyproject.toml", "CMakeLists.txt", "README.md", "PKG-INFO"}
    allowed_prefixes = ("cmake/", "src/analog_ecc_heights/", "tests/", "examples/", "docs/")
    unexpected = {
        name for name in names
        if name
        and name not in allowed_root_files
        and name != "src/solve_m_height_cpp.py"
        and not name.startswith(allowed_prefixes)
        and not ("/" not in name and name.startswith(("LICENSE", "COPYING", "NOTICE")))
    }
    assert not unexpected, f"Unexpected source archive contents: {sorted(unexpected)}"
    assert not any(
        part in {".git", ".github", "__pycache__", "build", ".pytest_cache"}
        for name in names for part in Path(name).parts
    )
