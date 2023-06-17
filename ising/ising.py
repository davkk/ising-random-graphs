import random
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit


def calc_E(*, edges: npt.NDArray[Any], spins: npt.NDArray[Any]) -> float:
    return np.sum(edges @ spins * spins) / 2.0


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

    energy = np.empty(steps)
    magnet = np.empty(steps)

    energy[0] = init_E
    magnet[0] = init_M

    seen = np.full_like(spins, False)

    for step in range(1, steps):
        energy[step] = energy[step - 1]
        magnet[step] = magnet[step - 1]

        while not np.all(seen):
            idx = np.random.choice(np.where(~seen)[0])
            seen[idx] = True

            spin_neighbors = np.sum(spins * edges[idx])

            dE = 2.0 * spins[idx] * spin_neighbors
            dM = -2.0 * spins[idx]

            if random.random() < min(np.exp(-dE * beta), 1) or dE < 0.0:
                spins[idx] = -spins[idx]

                energy[step] += dE
                magnet[step] += dM

    return beta, energy / N, magnet / N
