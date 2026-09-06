"""Compute heights with the default Python backend."""

import numpy as np

from analog_ecc_heights import (
    available_backends,
    h_m_jiang_simplified_lp,
    h_m_roth_dual_combinatorial_parity,
    h_m_roth_primal_combinatorial,
    h_m_roth_primal_lp,
)


def main():
    G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    H = np.array([[1.0, 1.0, -1.0]])
    print("Available backends:", available_backends())
    print("Roth LP h_1:", h_m_roth_primal_lp(G, 1))
    print("Jiang LP h_1, capped at 1.5:", h_m_jiang_simplified_lp(G, 1, early_quit_threshold=1.5))
    print("Primal combinatorial h_1:", h_m_roth_primal_combinatorial(G, 1))
    print("Parity combinatorial h_1:", h_m_roth_dual_combinatorial_parity(H, 1))
    print("Roth LP h_2:", h_m_roth_primal_lp(G, 2))


if __name__ == "__main__":
    main()
