import sys

import numpy as np
from networkx import barabasi_albert_graph, to_numpy_array
from numba import jit

np.random.seed(2001)


@jit(nopython=True, fastmath=True, cache=True)
def generate(*, n, k, edges):
    spins = np.random.choice(np.array([-1.0, 1.0]), size=n)

    if k > 1:
        layers = np.empty((k, edges.shape[0], edges.shape[1]))

        for ki in range(k):
            layers[ki] = np.linalg.matrix_power(edges, ki + 1)

        edges = (layers != 0).argmax(0)
        edges = np.where(edges != 0, np.exp(-edges), 0)

    return spins, edges


@jit(nopython=True, fastmath=True, cache=True)
def simulate(*, steps, spins, edges, temp):
    N = spins.shape[0]

    init_energy = -1.0 * np.sum(edges @ spins * spins) / 2.0
    init_magnet = np.sum(spins)

    energy = init_energy
    magnet = init_magnet

    for step in range(steps):
        idx = np.random.randint(N)

        spins[idx] *= -1.0  # change spin value
        new_energy = -1.0 * np.sum(np.dot(edges, spins) * spins) / 2.0

        dE = new_energy - energy
        dM = -2.0 * spins[idx]

        if dE < 0.0 or np.random.random() < np.exp(-dE / temp):
            energy += dE
            magnet += dM
        else:
            spins[idx] *= -1.0  # restore prev spin value, reject change

        print(temp, step, energy, magnet)


if __name__ == "__main__":
    spins, edges = generate(
        n=128,
        k=4,
        edges=to_numpy_array(
            barabasi_albert_graph(n=128, m=4),
            dtype=float,
        ),
    )

    simulate(
        temp=float(sys.argv[1]),
        steps=100_000,
        spins=spins,
        edges=edges,
    )
