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
        assert "Version: 0.1.0\n" in metadata
        assert "Requires-Python: >=3.10" in metadata
        native = any(
            name.startswith("analog_ecc_heights/cpp_backend/solve_m_height_cpp")
            and name.endswith((".so", ".pyd"))
            for name in names
        )
        if os.environ.get("ANALOG_ECC_EXPECT_HIGHS") is not None:
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
            name.startswith("analog_ecc_heights/cpp_backend/solve_m_height_cpp")
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
        if fullname == "solve_m_height_cpp" or fullname.endswith(".solve_m_height_cpp"):
            raise AssertionError("Default API attempted to load the native extension")

blocker = BlockNative()
sys.meta_path.insert(0, blocker)
import analog_ecc_heights as heights

assert Path(heights.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
G = [[1.0, 1.0, 1.0]]
assert abs(heights.h_m_roth_primal_lp(G, 1) - 1.0) < 1e-9
assert abs(heights.h_m_roth_primal_combinatorial(G, 1) - 1.0) < 1e-9
sys.meta_path.remove(blocker)

if sys.argv[2] == "native":
    from analog_ecc_heights.cpp_backend import _build_info
    assert abs(heights.h_m_roth_primal_lp(G, 1, backend="cpp-glpk") - 1.0) < 1e-9
    assert abs(heights.h_m_roth_primal_combinatorial(G, 1, backend="cpp") - 1.0) < 1e-9
    if _build_info.HAS_HIGHS:
        assert abs(heights.h_m_roth_primal_lp(G, 1, backend="cpp-highs") - 1.0) < 1e-9
    import solve_m_height_cpp as legacy
    assert abs(legacy.h_m_roth_primal_lp_glpk(G, 1, float("inf")) - 1.0) < 1e-9
    print(json.dumps({"highs": _build_info.HAS_HIGHS, "openmp": _build_info.HAS_OPENMP}))
else:
    try:
        heights.h_m_roth_primal_lp(G, 1, backend="cpp-glpk")
    except ImportError:
        pass
    else:
        raise AssertionError("A Python-only wheel unexpectedly provided cpp-glpk")
    print(json.dumps({"native": False}))
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
    expected_highs = os.environ.get("ANALOG_ECC_EXPECT_HIGHS")
    if native and expected_highs is not None:
        assert capabilities["highs"] is (expected_highs == "1")


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
        "src/analog_ecc_heights/cpp_backend/src/py_binding.cc",
        "src/analog_ecc_heights/cpp_backend/src/lp_workspace.hh",
        "src/analog_ecc_heights/cpp_backend/src/methods.hh",
        "src/analog_ecc_heights/cpp_backend/src/utils.hh",
        "src/analog_ecc_heights/cpp_backend/src/methods_jiang_lp.cc",
        "src/analog_ecc_heights/cpp_backend/src/methods_roth_lp.cc",
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
