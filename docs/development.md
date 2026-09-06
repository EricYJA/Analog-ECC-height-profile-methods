# Development and release preparation

Use the `analog-ecc` conda environment for this repository's code, tests, and build commands. Run commands below from the repository root unless stated otherwise. If conda has not been initialized in a Bash shell:

```bash
source /home/ccyuan/miniconda3/etc/profile.d/conda.sh
conda activate analog-ecc
```

On another machine, use that machine's conda initialization script and an environment named `analog-ecc`.

## Install and test

```bash
conda activate analog-ecc
python -m pip install ".[dev]"
python -m pytest -q
```

To test Python source changes directly without reinstalling:

```bash
conda activate analog-ecc
PYTHONPATH=src python -m pytest -q
```

The default suite checks public behavior, Python implementations, available-backend agreement, and numerical edge cases. Native comparisons run when the required extension is present. Artifact checks skip until the corresponding paths below are supplied. Existing native binding tests live under `tests/native/` and are excluded from the default collection:

```bash
conda activate analog-ecc
python -m pytest -q tests/native
```

Install a native-enabled package before running those tests. Use the [native installation instructions](installation.md), and test GLPK-only and HiGHS-enabled builds separately. Source-only `PYTHONPATH=src` runs should not be used to verify a wheel's compiled extension: the source tree can mask the installed package.

## Build release artifacts

The first release consists of the Python wheel and complete source distribution. A plain build keeps CMake disabled:

```bash
conda activate analog-ecc
python -m build --outdir dist
python -m twine check dist/*
```

For version 0.1.0, expect `analog_ecc_heights-0.1.0-py3-none-any.whl` and `analog_ecc_heights-0.1.0.tar.gz`. The source archive includes Python code, unchanged C++ sources, CMake configuration, tests, examples, and documentation. The Python wheel contains runtime Python files and the compatibility shim, without C++ source or compiled modules.

Validate actual artifacts rather than only imports from the checkout:

```bash
conda activate analog-ecc
python -m pip install --force-reinstall --no-deps dist/analog_ecc_heights-0.1.0-py3-none-any.whl
ANALOG_ECC_TEST_WHEEL=dist/analog_ecc_heights-0.1.0-py3-none-any.whl \
ANALOG_ECC_TEST_SDIST=dist/analog_ecc_heights-0.1.0.tar.gz \
  python -m pytest -q
```

The artifact tests inspect archive contents and wheel tags, install the wheel into a fresh environment outside the checkout, and check that default calls do not load native code. They reuse installed third-party dependencies, while CI provides fresh environments for dependency and installation checks.

Build a native wheel from the source archive to verify that the archive is complete:

```bash
conda activate analog-ecc
python -m pip wheel --no-deps --wheel-dir native-dist dist/analog_ecc_heights-0.1.0.tar.gz \
  -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON
```

Use `USE_HIGHS=OFF` for the second native configuration. Install each resulting wheel and repeat the tests with `ANALOG_ECC_TEST_WHEEL` pointing to that exact native wheel. Set `ANALOG_ECC_EXPECT_HIGHS=1` for HiGHS builds or `0` for GLPK-only builds. Keep native variants in separate output directories so they cannot overwrite each other; their filenames can be identical despite different solver features.

Native wheels built locally link against local system libraries. The first PyPI release uploads the default Python wheel and source archive, not those local validation wheels.

## Native build settings

Package builds default to Release, with these optional settings disabled:

| Setting | Purpose |
| --- | --- |
| `cmake.define.USE_HIGHS` | Require and enable system HiGHS |
| `cmake.define.BUILD_TEST_MAIN` | Build the standalone native test executable |
| `cmake.define.SANITIZE` | Enable supported address/undefined-behavior sanitizers |
| `cmake.define.USE_CUDA_FULL_SWEEP` | Unsupported in this package; keep `OFF` |

For a standalone native test build, install pybind11 into the active environment and configure CMake directly. Example for a standard GLPK/Eigen installation:

```bash
conda activate analog-ecc
python -m pip install "pybind11>=2.12"
cmake -S . -B build/native-tests \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DBUILD_TEST_MAIN=ON -DUSE_HIGHS=OFF
cmake --build build/native-tests --parallel 2
./build/native-tests/test_main
```

For HiGHS, enable `USE_HIGHS` and add its CMake prefix as needed. Compiler memory use can be substantial; `CMAKE_BUILD_PARALLEL_LEVEL=2` limits package-build compilation concurrency. The original computational sources under `src/analog_ecc_heights/cpp_backend/src/` should remain unchanged during this packaging migration.

## Continuous integration

`.github/workflows/tests.yml` builds and tests the default package on Linux, macOS, and Windows with Python 3.10 and 3.14. An additional Ubuntu/Python 3.10 job tests NumPy 1.23.0 and SciPy 1.9.0, the declared minimum versions. Native Linux jobs build both GLPK-only and HiGHS-enabled wheels from the source distribution and test their installed behavior. HiGHS is built as a system C++ library in the native job. CI validates artifacts but does not publish them.

## Prepare a PyPI release

Version 0.1.0 and the author entry, Changcheng Yuan (`eric.yuan.cc@gmail.com`), are configured in `pyproject.toml`. The project license has not been chosen. Add the selected license file and matching metadata before publication, and confirm the intended project URLs. Keep the version, examples, and artifact paths consistent when preparing later releases.

Build into a clean release-output directory, run the checks above, and review exactly the wheel and source archive intended for upload. The following commands are manual publishing instructions; builds and tests do not execute them:

```bash
conda activate analog-ecc
python -m twine upload --repository testpypi \
  dist/analog_ecc_heights-0.1.0-py3-none-any.whl \
  dist/analog_ecc_heights-0.1.0.tar.gz

# Publish the reviewed artifacts to PyPI when ready.
python -m twine upload \
  dist/analog_ecc_heights-0.1.0-py3-none-any.whl \
  dist/analog_ecc_heights-0.1.0.tar.gz
```

Configure publisher credentials outside repository files. TestPyPI and PyPI are separate services. See the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) for account setup and upload details.
