import random

import numpy as np
from igraph import Graph
from numpy.typing import NDArray


def simulate(
    *,
    steps: int,
    spins: NDArray[np.int_],
    graph: Graph,
    beta: float,
):
    print(f"{beta=:.3f}")

    neighbors = [
        np.sum(spins[graph.neighbors(i)]) for i in range(graph.vcount())
    ]

    energy = -spins @ neighbors / 2
    magnet = np.sum(spins)

    N = graph.vcount()

    for _ in range(steps):
        for idx, spin in enumerate(spins):
            neighbors = spins[graph.neighbors(idx)].sum()

            dE = 2 * spin * neighbors
            dM = -2 * spin

            if random.random() < min(np.exp(-dE * beta), 1) or dE < 0:
                spins[idx] = -spins[idx]

                energy += dE
                magnet += dM

    return beta, energy / N, magnet / N
