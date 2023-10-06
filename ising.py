import sys

import numpy as np
from numba import jit

np.random.seed(2001)


@jit(nopython=True, fastmath=True, cache=True)
def simulate(*, temp, steps, edges):
    n = edges.shape[0]
    spins = np.random.choice(np.array([-1.0, 1.0]), size=n)

    energy = -0.5 * np.sum(np.dot(edges, spins) * spins)
    magnet = np.sum(spins)

    for step in np.arange(1, steps + 1):
        idx = np.random.randint(n)

        spins[idx] *= -1.0  # change spin value
        new_energy = -0.5 * np.sum(np.dot(edges, spins) * spins)

        dE = new_energy - energy
        dM = -2.0 * spins[idx]

        if dE < 0.0 or np.random.random() < np.exp(-dE / temp):
            energy += dE
            magnet += dM
        else:
            spins[idx] *= -1.0  # restore prev spin value, reject change

        print(temp, step, energy, magnet)


simulate(
    edges=np.load(sys.argv[1]),
    steps=int(sys.argv[2]),
    temp=float(sys.argv[3]),
)
