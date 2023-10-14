import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC("ising")


@njit()
@cc.export("simulate", "Array(f8, 2, 'C'), i8, f8")
def simulate(edges, steps, temp):
    np.random.seed(2001)

    n = edges.shape[0]
    spins = np.random.choice(np.array([-1.0, 1.0]), size=n)

    energy = -0.5 * np.sum(np.dot(edges, spins) * spins)
    magnet = np.sum(spins)

    for step in np.arange(1, steps + 1):
        idx = np.random.randint(n)

        spins[idx] *= -1.0  # flip random spin
        new_energy = -0.5 * np.sum(np.dot(edges, spins) * spins)

        dE = new_energy - energy
        dM = -2.0 * spins[idx]

        if dE < 0.0 or np.random.random() < np.exp(-dE / temp):
            energy += dE
            magnet += dM
        else:
            spins[idx] *= -1.0  # restore spin value => reject the change

        print(temp, step, energy / n, magnet / n)


def main():
    cc.compile()


if __name__ == "__main__":
    main()
