# analog-ecc-heights

Python and C++ methods for computing the height profile of analog error-correcting codes, including Jiang's LP formulations and Roth's LP and combinatorial formulations. The distribution is **`analog-ecc-heights`**; the Python import is **`analog_ecc_heights`**.

One package provides a NumPy/SciPy implementation by default and optional native backends. Installing native support does not change the default backend.

## Install

Requires Python 3.10 or later, NumPy 1.23 or later, and SciPy 1.9 or later. From a checkout:

```bash
python -m pip install .
```

After the package is published to PyPI:

```bash
python -m pip install analog-ecc-heights
```

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
| `cpp` | Combinatorial | Eigen headers at build time; GLPK through the shared extension |
| `cpp-glpk` | LP | Eigen headers at build time; GLPK headers and library |
| `cpp-highs` | LP | Eigen headers at build time; GLPK and HiGHS headers and libraries |

An unavailable backend raises an installation error. Unsupported combinations raise an argument error. Calls never switch to another backend automatically.

## Documentation and examples

- [Installation and native builds](docs/installation.md)
- [API signatures, matrix restrictions, and return values](docs/api.md)
- [Backend behavior, numerical tolerances, and threading](docs/backends.md)
- [Development, tests, and release preparation](docs/development.md)
- Runnable examples: [basic usage](examples/basic_usage.py), [height profiles](examples/height_profile.py), and [backend comparison](examples/compare_backends.py)

The public API is in `src/analog_ecc_heights/`. Its `py_backend/` and `cpp_backend/` directories implement the same mathematical methods. The existing C++ computation and binding sources are preserved under `cpp_backend/src/`. Native-enabled installations retain `import solve_m_height_cpp` for existing callers, within the same distribution.

Version: **0.1.0**. Author: **Changcheng Yuan** (`eric.yuan.cc@gmail.com`). A project license has not yet been specified; release preparation must include the selected license and corresponding metadata.

## References

All implementations are based on the methods and theorems developed in these papers:

1. Ron M. Roth, “[Analog Error-Correcting Codes](https://doi.org/10.1109/TIT.2020.2977918),” *IEEE Transactions on Information Theory*, 66(7), 4075–4088, 2020.
2. Anxiao Jiang, “[Analog Error-Correcting Codes: Designs and Analysis](https://doi.org/10.1109/TIT.2024.3454059),” *IEEE Transactions on Information Theory*, 70(11), 7740–7756, 2024.
3. Ron M. Roth, Ziyuan Zhu, Changcheng Yuan, Paul H. Siegel, and Anxiao Jiang, “[On the Height Profile of Analog Error-Correcting Codes](https://arxiv.org/abs/2602.20366),” *2026 IEEE International Symposium on Information Theory (ISIT)*, also available as arXiv:2602.20366.
