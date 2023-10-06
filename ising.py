import numpy as np
from numba import float32, njit
from numba.pycc import CC

cc = CC("ising")
np.random.seed(2001)


@njit()
@cc.export("simulate", "Array(f4, 2, 'C'), i8, f4")
def simulate(edges, steps, temp):
    n = edges.shape[0]
    spins = np.random.choice(np.array([-1.0, 1.0], dtype=float32), size=n)

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


if __name__ == "__main__":
    cc.compile()
