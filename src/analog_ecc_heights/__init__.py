"""LP and combinatorial heights of analog error-correcting codes."""

from importlib.metadata import PackageNotFoundError, version

from ._dispatch import available_backends
from .combinatorial import (
    h_m_roth_dual_combinatorial_generator,
    h_m_roth_dual_combinatorial_parity,
    h_m_roth_mds_combinatorial,
    h_m_roth_mds_combinatorial_parity,
    h_m_roth_primal_combinatorial,
    h_m_roth_primal_combinatorial_pruning,
)
from .lp import (
    h_m_jiang_original_lp,
    h_m_jiang_simplified_lp,
    h_m_roth_dual_lp,
    h_m_roth_primal_lp,
)

try:
    __version__ = version("analog-ecc-heights")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "available_backends", "__version__",
    "h_m_jiang_original_lp", "h_m_jiang_simplified_lp",
    "h_m_roth_primal_lp", "h_m_roth_dual_lp",
    "h_m_roth_primal_combinatorial", "h_m_roth_primal_combinatorial_pruning",
    "h_m_roth_dual_combinatorial_generator", "h_m_roth_dual_combinatorial_parity",
    "h_m_roth_mds_combinatorial", "h_m_roth_mds_combinatorial_parity",
]
