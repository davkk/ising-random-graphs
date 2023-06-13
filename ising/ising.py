import random
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit


def calc_E(
    *, interact: int, edges: npt.NDArray[Any], spins: npt.NDArray[Any]
) -> float:
    layers = np.empty((interact, edges.shape[0], edges.shape[1]))

    for i in range(interact):
        layers[i] = np.linalg.matrix_power(edges, i + 1)

    J_matrix = np.amax(layers, axis=0)

    spins = J_matrix @ spins

    neighbors = np.array(
        [np.sum(spins[np.where(edges[i])]) for i in range(spins.shape[0])]
    )

    return -spins @ neighbors / 2


def calc_M(*, spins: npt.NDArray[Any]) -> float:
    return np.sum(spins)


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def simulate(
    *,
    steps: int,
    beta: float,
    spins: npt.NDArray[Any],
    edges: npt.NDArray[Any],
    init_E: float,
    init_M: float,
    num_repeat: int,
):
    print(num_repeat + 1, ":", beta)

    N = float(spins.shape[0])

    energy = init_E
    magnet = init_M

    for _ in range(steps):
        for idx, spin in enumerate(spins):
            spin_neighbors = np.sum(spins[np.where(edges[idx])])

            dE = 2.0 * spin * spin_neighbors
            dM = -2.0 * spin

            if random.random() < min(np.exp(-dE * beta), 1) or dE < 0.0:
                spins[idx] = -spins[idx]

                energy += dE
                magnet += dM

    return beta, energy / N, magnet / N
