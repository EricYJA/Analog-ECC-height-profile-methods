# Installation

`analog-ecc-heights` provides a Python backend by default and optional C++ backends. Import it as `analog_ecc_heights`. Install the backends you need, then select one with the API's [`backend=` argument](api.md#backend-availability-and-threads).

## Choose a Python environment

Requires Python >=3.10. Use an existing environment or create one with Python's built-in `venv`:

```bash
python -m venv .venv
```

Use `python3` or `py` instead if that is how you invoke Python >=3.10 on your system. Activate the environment with the command for your shell:

| Platform / shell | Activation command |
| --- | --- |
| Linux or macOS / Bash or Zsh | `source .venv/bin/activate` |
| Windows / PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows / Command Prompt | `.venv\Scripts\activate.bat` |

After activation, `python` selects the environment's interpreter. If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of `python` in subsequent commands. See the [Python virtual environment documentation](https://docs.python.org/3/library/venv.html) for other shells and activation details.

### Optional Conda environment

If you use Conda, create and activate an environment instead of creating a `venv`:

```bash
conda create --name ecc-heights "python>=3.10" pip
conda activate ecc-heights
```

`ecc-heights` is an example name; choose any name or activate an existing environment. If activation is unavailable, initialize your shell with `conda init bash` or `conda init powershell`, as appropriate, and reopen the terminal. See [Conda shell initialization](https://docs.conda.io/projects/conda/en/stable/commands/init.html). No particular Conda installation path is required.

## Default Python installation

With your chosen environment active, install the published package from [PyPI](https://pypi.org/project/analog-ecc-heights/):

```bash
python -m pip install analog-ecc-heights
```

To install this release explicitly:

```bash
python -m pip install analog-ecc-heights==0.1.2
```

To upgrade an existing installation to the latest release:

```bash
python -m pip install --upgrade analog-ecc-heights
```

The [0.1.2 release](https://pypi.org/project/analog-ecc-heights/0.1.2/#files) includes a Python wheel and source archive. Its wheel provides the Python backend; optional native backends require a source build.

Pip installs NumPy >=1.23 and SciPy >=1.9 as runtime dependencies. This installs the Python backend without requiring a C++ compiler, CMake, Eigen, or system solver libraries.

SciPy supplies its own LP implementation using HiGHS. A separate HiGHS installation and the `highspy` package are unnecessary for this backend.

## Get the source

For a local source installation or to work on the code, install Git and clone the repository into a directory of your choice:

```bash
git clone https://github.com/EricYJA/Analog-ECC-height-profile-methods.git
cd Analog-ECC-height-profile-methods
```

If you already have a checkout or an extracted source distribution, open a terminal in its root directory, alongside `pyproject.toml`. Commands using `pip install .` run from this source root.

With your chosen Python environment active, install the checkout with:

```bash
python -m pip install .
```

This also defaults to the Python backend. To enable C++ backends, use the build settings below.

## Native requirements

Installing a native backend from source needs a C++17 compiler, CMake >=3.18, Python development headers, and Eigen headers. Pip installs the Python build tools automatically.

| API backend | Additional requirements |
| --- | --- |
| `cpp` | OpenMP |
| `cpp-glpk` | System GLPK development headers and library |
| `cpp-highs` | System HiGHS development headers and library; OpenMP |

Each backend requires only its own dependencies. Eigen is header-only and needed at build time. Solver libraries and compiler/OpenMP runtimes must remain available after installation. Installing only a solver executable or Python wrapper does not provide the development files required here.

OpenMP is mandatory for combinatorial and HiGHS builds: configuration fails if it cannot be found. GLPK-only and Python-only installations do not require OpenMP. Passing `num_threads=1` still selects sequential execution in a parallel-capable extension.

For example, on Debian/Ubuntu install the base native prerequisites with:

```bash
sudo apt-get install libeigen3-dev cmake ninja-build g++ python3-dev
# Add this only when enabling GLPK:
sudo apt-get install libglpk-dev
```

For HiGHS, install its C++ development library, headers, and preferably its CMake package configuration; see the [HiGHS installation guide](https://ergo-code.github.io/HiGHS/dev/installation/). The `highspy` Python package alone is not the supported dependency. Compilers such as Apple Clang may require a separately installed OpenMP runtime and explicit search paths.

## Install optional native backends

First install the [native requirements](#native-requirements) and activate your Python environment. Upgrade pip so it supports the `-C` build options used below:

```bash
python -m pip install --upgrade pip
```

Multiline commands use Bash syntax; in PowerShell or Command Prompt, put each command on one line and omit the trailing `\` characters.

### Build the published release from PyPI

For combinatorial support, build the published source archive with:

```bash
python -m pip install --force-reinstall --no-cache-dir \
  --no-binary=analog-ecc-heights analog-ecc-heights==0.1.2 \
  -Cwheel.cmake=true
```

`--no-binary=analog-ecc-heights` selects this package's source archive while allowing wheels for NumPy and SciPy. `--force-reinstall` rebuilds even if the same version is already installed, and `--no-cache-dir` avoids reusing a wheel from an earlier build. If NumPy and SciPy already satisfy the requirements, add `--no-deps` to keep those dependencies unchanged. See the [pip install options](https://pip.pypa.io/en/stable/cli/pip_install/#options).

Add the CMake flags from the configurations below to enable GLPK, HiGHS, or all three extensions when building from PyPI.

### Build from a local checkout or extracted source archive

Run from the [source root](#get-the-source) and choose one configuration:

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

To replace an existing installation of the same version, add `--force-reinstall --no-deps --no-cache-dir` to the chosen command. `--no-deps` assumes NumPy and SciPy are already installed. Native features are selected for the entire installation: rebuilding replaces the previous selection rather than adding to it. For a local source build, reinstalling with `-Cwheel.cmake=false` returns to Python-only support. To switch back to the published Python wheel, with NumPy and SciPy already installed, run:

```bash
python -m pip install --force-reinstall --no-deps --no-cache-dir \
  --only-binary=analog-ecc-heights analog-ecc-heights==0.1.2
```

## Nonstandard library locations

The examples in this section run from the local source root. Only set additional search paths when CMake cannot find dependencies in standard locations. Replace the example prefix below with the directory containing your dependency's `include` and `lib` directories.

In Bash:

```bash
CMAKE_PREFIX_PATH="/path/to/native/prefix" \
  python -m pip install . -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON
```

In PowerShell:

```powershell
$env:CMAKE_PREFIX_PATH = "C:/path/to/native/prefix"
python -m pip install . -Cwheel.cmake=true -Ccmake.define.USE_HIGHS=ON
```

For dependencies installed in an active Conda environment, its prefix is available as `$CONDA_PREFIX` in Bash or `$env:CONDA_PREFIX` in PowerShell. These examples specify one prefix; retain any other search paths your build needs.

Use the `CMAKE_PREFIX_PATH` environment variable here. Passing `-Ccmake.define.CMAKE_PREFIX_PATH=...` replaces the paths that scikit-build-core supplies for isolated build dependencies such as pybind11.

The CMake search also accepts these explicit settings:

| Setting | Value |
| --- | --- |
| `cmake.define.Eigen3_DIR` | Directory containing `Eigen3Config.cmake` |
| `cmake.define.GLPK_INCLUDE_DIR` | Directory containing `glpk.h` |
| `cmake.define.GLPK_LIBRARY` | Full path to the GLPK library |
| `cmake.define.HIGHS_INCLUDE_ROOT` | Include root containing `highs/Highs.h` |
| `cmake.define.HIGHS_LIBRARY` | Full path to the HiGHS library |

Keep native libraries and their runtimes available after installation. If a backend cannot be imported, the error includes the original loader message and suggested installation flags. Check that its libraries are installed and discoverable by your operating system.

## Migrate an older native installation

If the old `solve-m-height-cpp` distribution is installed, remove it before installing the new native-enabled package:

```bash
python -m pip uninstall solve-m-height-cpp
```

Then follow the native build instructions above. An older top-level `solve_m_height_cpp` compiled module can shadow the new compatibility shim in the same environment. The new `analog_ecc_heights` public import uses its own packaged extension and is unaffected, but removing the obsolete distribution keeps legacy imports consistent. This removal is only needed when migrating that old installation.

## Check an installation

```python
from analog_ecc_heights import __version__, available_backends, h_m_roth_primal_lp

print(__version__)
print(available_backends())  # ["python"] for the published Python wheel
print(h_m_roth_primal_lp([[1.0, -2.0, 4.0]], 1))  # 2.0, always Python by default
```

`available_backends()` checks each native extension independently. Calling `backend="cpp-highs"` cannot compile or install it on demand. An unavailable backend raises `ImportError`; installing native support never changes the default from `"python"`.
