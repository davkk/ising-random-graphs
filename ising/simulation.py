import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC("ising")


@njit(parallel=True, fastmath=True, cache=True)
@cc.export("simulate", "Array(f8, 2, 'C'), i8, f8")
def simulate(graph, steps, temp):
    n = graph.shape[0]
    spins = np.random.choice(np.array([-1.0, 1.0]), size=n)

    energy = -0.5 * np.sum(np.dot(graph, spins) * spins)
    magnet = np.sum(spins)

    for step in np.arange(1, steps + 1):
        idx = np.random.randint(n)

        spin = spins[idx]
        edges = graph[idx]
        neighbors = np.dot(edges, spins) - edges[idx] * spin

        dE = 2.0 * neighbors * spin
        dM = -2.0 * spin

        if dE < 0.0 or np.random.random() < np.exp(-dE / temp):
            spins[idx] *= -1.0
            energy += dE
            magnet += dM

        print(temp, step, energy / n, magnet / n)


def main():
    cc.compile()


if __name__ == "__main__":
    main()
