# analog-ecc-heights

Python and C++ methods for computing the height profile of analog error-correcting codes, including Jiang's LP formulations and Roth's LP and combinatorial formulations. The distribution is **`analog-ecc-heights`**; the Python import is **`analog_ecc_heights`**.

One package provides a NumPy/SciPy implementation by default and optional native backends. Installing native support does not change the default backend.

## Install

Requires Python 3.10 or later. Install the published package from [PyPI](https://pypi.org/project/analog-ecc-heights/):

```bash
python -m pip install analog-ecc-heights
```

Pip installs NumPy >=1.23 and SciPy >=1.9 automatically. Version [0.1.1](https://pypi.org/project/analog-ecc-heights/0.1.1/) is available as a Python wheel and source archive.

Use an existing Python environment or follow the [environment setup guide](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/docs/installation.md#choose-a-python-environment) to create a virtual environment or use Conda. If your Python command is `python3` or `py`, use it in place of `python`.

The default installation does not invoke CMake or require a compiler, Eigen, GLPK, or a separately installed HiGHS library. Native support is an optional source build of this same distribution; see [native installation](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/docs/installation.md#install-optional-native-backends). To install from a checkout, see [source installation](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/docs/installation.md#get-the-source).

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

With `m=None`, the primal combinatorial method returns the finite profile `[h_1, ..., h_(d-1)]`, where `d` is the minimum distance under `tol`. Its length is `d-1`; codes with `d=1` return `[]`.

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

- [Installation and native builds](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/docs/installation.md)
- [API, backend selection, and numerical behavior](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/docs/api.md)
- Runnable examples: [basic usage](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/examples/basic_usage.py), [height profiles](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/examples/height_profile.py), and [backend comparison](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/examples/compare_backends.py)

## License

Licensed under the [MIT License](https://github.com/EricYJA/Analog-ECC-height-profile-methods/blob/main/LICENSE).

## References

All implementations are based on the methods and theorems developed in these papers:

1. Ron M. Roth, “[Analog Error-Correcting Codes](https://doi.org/10.1109/TIT.2020.2977918),” *IEEE Transactions on Information Theory*, 66(7), 4075–4088, 2020.
2. Anxiao Jiang, “[Analog Error-Correcting Codes: Designs and Analysis](https://doi.org/10.1109/TIT.2024.3454059),” *IEEE Transactions on Information Theory*, 70(11), 7740–7756, 2024.
3. Ron M. Roth, Ziyuan Zhu, Changcheng Yuan, Paul H. Siegel, and Anxiao Jiang, “[On the Height Profile of Analog Error-Correcting Codes](https://arxiv.org/abs/2602.20366),” *2026 IEEE International Symposium on Information Theory (ISIT)*, also available as arXiv:2602.20366.
