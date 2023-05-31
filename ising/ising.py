import random

import numpy as np

from .domain import Lattice, Parameters


def simulate(
    *,
    params: Parameters,
    lattice: Lattice,
    num_repeat: int,
):
    print(f"{num_repeat + 1}: beta={params.beta}")

    N = lattice.graph.vcount()

    energy = lattice.initial_E
    magnet = lattice.initial_M

    for _ in range(params.steps):
        for idx, spin in enumerate(lattice.spins):
            spin_neighbors = np.sum(lattice.spins[lattice.graph.neighbors(idx)])

            dE = 2 * spin * spin_neighbors
            dM = -2 * spin

            if random.random() < min(np.exp(-dE * params.beta), 1) or dE < 0:
                lattice.spins[idx] = -lattice.spins[idx]

                energy += dE
                magnet += dM

    return params.beta, energy / N, magnet / N
