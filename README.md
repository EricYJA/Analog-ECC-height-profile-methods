# analog-ecc-heights

Python and C++ methods for computing the height profile of analog error-correcting codes, including Jiang's LP formulations and Roth's LP and combinatorial formulations. The distribution is **`analog-ecc-heights`**; the Python import is **`analog_ecc_heights`**.

One package provides a NumPy/SciPy implementation by default and optional native backends. Installing native support does not change the default backend.

## Install

Requires Python 3.10 or later. Pip installs NumPy >=1.23 and SciPy >=1.9 automatically. With Python and Git installed, start from a fresh checkout:

```bash
git clone https://github.com/EricYJA/Analog-ECC-height-profile-methods.git
cd Analog-ECC-height-profile-methods
python -m venv .venv
```

If your Python command is `python3` or `py`, use it to create the environment. Activate it using the command for your shell:

| Platform / shell | Activation command |
| --- | --- |
| Linux or macOS / Bash or Zsh | `source .venv/bin/activate` |
| Windows / PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows / Command Prompt | `.venv\Scripts\activate.bat` |

Then install into the active environment:

```bash
python -m pip install .
```

An existing Python environment also works. See [environment setup](docs/installation.md#choose-a-python-environment) for details and the optional Conda workflow.

The default installation does not invoke CMake or require a compiler, Eigen, GLPK, or a separately installed HiGHS library. Native support is an optional source build of this same distribution; see [installation](docs/installation.md).

## Use

```python
import numpy as np
from analog_ecc_heights import (
    available_backends,
    h_m_roth_primal_lp,
    h_m_roth_primal_combinatorial,
)

G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

print(h_m_roth_primal_lp(G, 1))                         # 2.0
print(h_m_roth_primal_lp(G, 1, early_quit_threshold=1.5)) # 1.5
print(h_m_roth_primal_combinatorial(G))                # [2.0]
print(available_backends())                           # ["python"] in a default install
```

For a native-enabled installation, select the implementation explicitly:

```python
h_m_roth_primal_lp(G, 1, backend="cpp-glpk")
h_m_roth_primal_lp(G, 1, backend="cpp-highs", num_threads=1)
h_m_roth_primal_combinatorial(G, 1, backend="cpp", num_threads=1)
```

| Backend | Methods | Separately installed native dependencies |
| --- | --- | --- |
| `python` | LP and combinatorial | None; LP uses the HiGHS solver included with SciPy |
| `cpp` | Combinatorial | Eigen headers at build time; OpenMP |
| `cpp-glpk` | LP | Eigen headers at build time; GLPK headers and library |
| `cpp-highs` | LP | Eigen headers at build time; HiGHS headers and library; OpenMP |

An unavailable backend raises an installation error. Unsupported combinations raise an argument error. Calls never switch to another backend automatically.

## Documentation and examples

- [Installation and native builds](docs/installation.md)
- [API, backend selection, and numerical behavior](docs/api.md)
- Runnable examples: [basic usage](examples/basic_usage.py), [height profiles](examples/height_profile.py), and [backend comparison](examples/compare_backends.py)

## References

All implementations are based on the methods and theorems developed in these papers:

1. Ron M. Roth, “[Analog Error-Correcting Codes](https://doi.org/10.1109/TIT.2020.2977918),” *IEEE Transactions on Information Theory*, 66(7), 4075–4088, 2020.
2. Anxiao Jiang, “[Analog Error-Correcting Codes: Designs and Analysis](https://doi.org/10.1109/TIT.2024.3454059),” *IEEE Transactions on Information Theory*, 70(11), 7740–7756, 2024.
3. Ron M. Roth, Ziyuan Zhu, Changcheng Yuan, Paul H. Siegel, and Anxiao Jiang, “[On the Height Profile of Analog Error-Correcting Codes](https://arxiv.org/abs/2602.20366),” *2026 IEEE International Symposium on Information Theory (ISIT)*, also available as arXiv:2602.20366.
