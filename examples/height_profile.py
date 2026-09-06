"""A full profile includes h_1 through h_(n-k), without h_0."""

import numpy as np

from analog_ecc_heights import h_m_roth_primal_combinatorial


def main():
    G = np.array([[1.0, -2.0, 4.0]])
    profile = h_m_roth_primal_combinatorial(G)
    print("Profile:", profile)
    for m, height in enumerate(profile, start=1):
        print(f"h_{m} = {height:g}")
    print("Scalar h_0:", h_m_roth_primal_combinatorial(G, 0))


if __name__ == "__main__":
    main()
