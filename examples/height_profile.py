"""A finite profile includes h_1 through h_(d-1), using minimum distance d."""

import numpy as np

from analog_ecc_heights import h_m_roth_primal_combinatorial


def main():
    G = np.array([[1.0, -2.0, 4.0]])
    profile = h_m_roth_primal_combinatorial(G)
    print("Profile:", profile)
    for m, height in enumerate(profile, start=1):
        print(f"h_{m} = {height:g}")
    print("Scalar h_0:", h_m_roth_primal_combinatorial(G, 0))

    G_distance_two = np.array([[0.0, 1.0, 2.0]])
    print("Distance-2 profile:", h_m_roth_primal_combinatorial(G_distance_two))
    print("Scalar h_2:", h_m_roth_primal_combinatorial(G_distance_two, 2))


if __name__ == "__main__":
    main()
