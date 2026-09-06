# Installation

`analog-ecc-heights` is one distribution with one version and the import name `analog_ecc_heights`. It supports a default Python installation and three independently optional native extensions. Select native support while installing; select an installed implementation with the API's `backend=` argument.

## Default Python installation

Requires Python >=3.10, NumPy >=1.23, and SciPy >=1.9. From this repository's root:

```bash
python -m pip install .
```

Once published, install the same package from PyPI:

```bash
python -m pip install analog-ecc-heights
```

Both commands default to the Python backend. The package's Python wheel contains no compiled extension, and the default source build has `wheel.cmake=false`: it does not invoke CMake or require a C++ compiler or system solver libraries. Source builds install their Python build tools in pip's isolated build environment.

SciPy supplies its own LP implementation using HiGHS. A separate HiGHS installation and the `highspy` package are unnecessary for this backend.

## Native requirements

Every native build needs a C++17 compiler, CMake >=3.18, Python development headers, and Eigen headers. Pip installs scikit-build-core and pybind11 as build dependencies.

| Extension | API backend | Additional requirements |
| --- | --- | --- |
| `_comb` | `cpp` | OpenMP |
| `_glpk` | `cpp-glpk` | System GLPK development headers and library |
| `_highs` | `cpp-highs` | System HiGHS development headers and library; OpenMP |

Each extension links only its own dependencies. Eigen is header-only and needed at build time. Solver libraries and compiler/OpenMP runtimes must remain available after installation. Installing only a solver executable or Python wrapper does not provide the development files required here.

OpenMP is mandatory for combinatorial and HiGHS builds: configuration fails if it cannot be found. GLPK-only and Python-only installations do not require OpenMP. Passing `num_threads=1` still selects sequential execution in a parallel-capable extension.

For example, on Debian/Ubuntu install the base native prerequisites with:

```bash
sudo apt-get install libeigen3-dev cmake ninja-build g++ python3-dev
# Add this only when enabling GLPK:
sudo apt-get install libglpk-dev
```

For HiGHS, install its C++ development library, headers, and preferably its CMake package configuration; see the [HiGHS installation guide](https://ergo-code.github.io/HiGHS/dev/installation/). CI builds HiGHS 1.11.0 from source. The `highspy` Python package alone is not the supported dependency. Compilers such as Apple Clang may require a separately installed OpenMP runtime and explicit search paths.

## Build native support from this checkout

Run from the repository root after installing the dependencies:

```bash
# Combinatorial only (the default native configuration)
python -m pip install . -Cwheel.cmake=true

# GLPK only; no OpenMP requirement
python -m pip install . -Cwheel.cmake=true \
  -Ccmake.define.BUILD_CPP_COMB=OFF -Ccmake.define.USE_GLPK=ON

# HiGHS only
python -m pip install . -Cwheel.cmake=true \
  -Ccmake.define.BUILD_CPP_COMB=OFF -Ccmake.define.USE_HIGHS=ON

# All three native extensions
python -m pip install . -Cwheel.cmake=true \
  -Ccmake.define.USE_GLPK=ON -Ccmake.define.USE_HIGHS=ON
```

All variants include the default Python implementation. Build options default to `BUILD_CPP_COMB=ON`, `USE_GLPK=OFF`, and `USE_HIGHS=OFF`. Combine them as needed. CMake runs only with `wheel.cmake=true`; every requested dependency must be found. Selecting no native extension with CMake enabled raises a configuration error.

To replace an existing installation of the same version, add `--force-reinstall --no-deps --no-cache-dir` to the chosen command. `--no-deps` assumes NumPy and SciPy are already installed. Native features are selected for the entire installation: rebuilding replaces the previous selection rather than adding to it. Reinstalling with `-Cwheel.cmake=false` returns to Python-only support.

## Build native support from a published release

After publication, force a source build of this package to enable native support. For example, to install all three extensions:

```bash
python -m pip install "numpy>=1.23" "scipy>=1.9"
python -m pip install analog-ecc-heights --force-reinstall --no-deps --no-cache-dir \
  --no-binary=analog-ecc-heights -Cwheel.cmake=true \
  -Ccmake.define.USE_GLPK=ON -Ccmake.define.USE_HIGHS=ON
```

Use the same build options as above for other combinations. `--no-cache-dir` avoids reusing a wheel built with different options. Normal upgrades install the default wheel unless a source build is requested again.

## Nonstandard library locations

Point CMake at an installation prefix when dependencies are installed outside standard search locations:

```bash
python -m pip install . -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON \
  -Ccmake.define.CMAKE_PREFIX_PATH=/path/to/native/prefix
```

The CMake search also accepts these explicit settings:

| Setting | Value |
| --- | --- |
| `cmake.define.Eigen3_DIR` | Directory containing `Eigen3Config.cmake` |
| `cmake.define.GLPK_INCLUDE_DIR` | Directory containing `glpk.h` |
| `cmake.define.GLPK_LIBRARY` | Full path to the GLPK library |
| `cmake.define.HIGHS_INCLUDE_ROOT` | Include root containing `highs/Highs.h` |
| `cmake.define.HIGHS_LIBRARY` | Full path to the HiGHS library |

A locally compiled wheel links to the installed native libraries; it is not a portable, bundled native release. Keep those libraries available after installation. If import fails, the raised error includes the original loader message and suggested build flags. Native Linux builds are covered by CI; other native toolchains may need explicit dependency and compiler paths.

## Migrate an older native installation

If the old `solve-m-height-cpp` distribution is installed, remove it before installing the new native-enabled package:

```bash
python -m pip uninstall solve-m-height-cpp
```

Then follow the native build instructions above. An older top-level `solve_m_height_cpp` compiled module can shadow the new compatibility shim in the same environment. The new `analog_ecc_heights` public import uses its own packaged extension and is unaffected, but removing the obsolete distribution keeps legacy imports consistent. This removal is only needed when migrating that old installation.

## Check an installation

```python
from analog_ecc_heights import available_backends, h_m_roth_primal_lp

print(available_backends())
print(h_m_roth_primal_lp([[1.0, -2.0, 4.0]], 1))  # 2.0, always Python by default
```

`available_backends()` checks each native extension independently. Calling `backend="cpp-highs"` cannot compile or install it on demand. An unavailable backend raises `ImportError`; installing native support never changes the default from `"python"`.
