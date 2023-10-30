import numba as nb
import numpy as np
from numba.pycc import CC

cc = CC("ising")


@nb.njit(parallel=True, fastmath=True, cache=True)
@cc.export("simulate", "Array(f8, 2, 'C'), i8, f8")
def simulate(graph, steps, temp):
    n = graph.shape[0]
    spins = np.random.choice(np.array([-1.0, 1.0]), size=n)

    energy = -0.5 * np.sum(np.dot(graph, spins) * spins)
    magnet = np.sum(spins)

    for step in range(1, steps * n + 1):
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

        if step % n == 0:
            print(temp, step // n, energy / n, magnet / n)


@nb.njit(parallel=True, fastmath=True, cache=True)
@cc.export("shortpath", "Array(f8, 2, 'C')(Array(u1, 2, 'C'))")
def shortpath(graph):
    n = graph.shape[0]

    J = np.zeros(n * n, dtype=np.float64)

    q = np.full(n, -1, dtype=np.uint8)

    for node in range(n):
        path_len = np.zeros(n, np.uint16)
        path_len[node] = 0

        front, rear = 0, 1
        q[front] = node

        while front < rear:
            curr = q[front]
            front += 1

            J[node * n + curr] = path_len[curr]

            for child in range(n):
                if graph[curr, child] == 0:
                    continue

                if path_len[child] > 0:
                    continue

                path_len[child] = path_len[curr] + 1

                q[rear] = child
                rear += 1

    J = J.reshape(n, n)
    J = np.exp(-(J - 1))
    np.fill_diagonal(J, 0)
    return J


def main():
    cc.compile()


if __name__ == "__main__":
    main()
