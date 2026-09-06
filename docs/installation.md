# Installation

`analog-ecc-heights` is one distribution with one version and the import name `analog_ecc_heights`. It supports a default Python installation and two optional native build configurations. Select native support while installing; select an installed implementation with the API's `backend=` argument.

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

Every native build needs a C++17 compiler, CMake >=3.18, Python development headers, Eigen headers, and the GLPK development headers and library. Pip installs scikit-build-core and pybind11 as build dependencies. OpenMP is optional.

| Build configuration | Available backends | Required solver development libraries |
| --- | --- | --- |
| Default | `python` | None |
| Native, `USE_HIGHS=OFF` | `python`, `cpp`, `cpp-glpk` | GLPK |
| Native, `USE_HIGHS=ON` | `python`, `cpp`, `cpp-glpk`, `cpp-highs` | GLPK and HiGHS |

The existing C++ sources form one extension, so native combinatorial and HiGHS builds also link GLPK. Eigen is header-only and is needed when compiling, not as a runtime installation. Shared solver libraries and the compiler/OpenMP runtime must remain discoverable when loading the built extension. Installing only a solver executable or a Python wrapper does not provide the development files required here.

For example, on Debian/Ubuntu the GLPK/Eigen build prerequisites can be installed with:

```bash
sudo apt-get install libeigen3-dev libglpk-dev cmake ninja-build g++ python3-dev
```

For HiGHS, install its C++ development library, headers, and preferably its CMake package configuration; follow the [HiGHS installation guide](https://ergo-code.github.io/HiGHS/dev/installation/). The native CI configuration builds HiGHS 1.11.0 from source. `pip install highspy` alone is not the supported way to satisfy this dependency.

## Build native support from this checkout

Run from the repository root, after installing the system dependencies:

```bash
# GLPK and C++ combinatorial support
python -m pip install . -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=OFF

# GLPK, HiGHS, and C++ combinatorial support
python -m pip install . -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON
```

CMake runs only when `wheel.cmake=true`. Enabling HiGHS requires it to be found; the build fails clearly if its development library is missing. The flags are passed through pip's `--config-settings` (`-C`) interface; see [scikit-build-core configuration](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html).

If an existing installation must be replaced without upgrading NumPy or SciPy, use:

```bash
python -m pip install . --force-reinstall --no-deps --no-cache-dir \
  -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON
```

`--no-deps` assumes the runtime dependencies are already installed. Change `USE_HIGHS` to `OFF` for the GLPK-only configuration. Reinstalling with `-Cwheel.cmake=false` returns to a Python-only installation.

## Build native support from a published release

The first release format is a Python wheel plus a complete source archive. To enable native support, force a source build of this package rather than selecting its default wheel. These commands apply after publication:

```bash
python -m pip install "numpy>=1.23" "scipy>=1.9"
python -m pip install analog-ecc-heights --force-reinstall --no-deps --no-cache-dir \
  --no-binary=analog-ecc-heights \
  -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON
```

Again, use `USE_HIGHS=OFF` for GLPK-only support. `--force-reinstall` matters when the same version is already installed; `--no-cache-dir` avoids reusing a wheel built with different native options. Normal upgrades install the default wheel unless a source build is requested again.

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

`available_backends()` checks whether the native extension can be loaded. Calling `backend="cpp-highs"` cannot compile or install it on demand. An unavailable backend raises `ImportError`; installing native support never changes the default from `"python"`.
