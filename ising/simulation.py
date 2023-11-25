import numba as nb
import numpy as np
from numba.pycc import CC

cc = CC("ising")


@nb.njit(parallel=True, cache=True)
@cc.export("simulate", "Array(f8, 2, 'C'), i8, f8")
def simulate(graph, steps, temp):
    n = graph.shape[0]
    spins = np.random.choice(np.array([-1.0, 1.0]), size=n)

    energy = -0.5 * np.sum((graph @ spins) * spins)
    magnet = np.sum(spins)

    for step in range(1, steps * n + 1):
        idx = np.random.randint(n)

        spin = spins[idx]
        neighbors = graph[idx] @ spins

        dE = 2.0 * neighbors * spin
        dM = -2.0 * spin

        if dE < 0.0 or np.random.random() < np.exp(-dE / temp):
            spins[idx] *= -1.0
            energy += dE
            magnet += dM

        if step % n == 0:
            print(temp, step // n, energy / n, abs(magnet) / n)


def main():
    cc.compile()


if __name__ == "__main__":
    main()
