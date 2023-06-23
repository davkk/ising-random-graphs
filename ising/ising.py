import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit


def calc_E(*, edges: npt.NDArray[Any], spins: npt.NDArray[Any]) -> float:
    return -np.sum(edges @ spins * spins) / 2.0


def calc_M(*, spins: npt.NDArray[Any]) -> float:
    return np.sum(spins)


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def simulate(
    *,
    steps: int,
    temp: float,
    spins: npt.NDArray[Any],
    edges: npt.NDArray[Any],
    init_E: float,
    init_M: float,
    num_repeat: int,
):
    print(num_repeat + 1, ":", temp)

    N = spins.shape[0]

    energies = np.empty(steps)
    magnets = np.empty(steps)

    energy = init_E
    magnet = init_M

    for step in range(0, steps):
        idx = np.random.randint(N)

        spin_neighbors = np.sum(spins * edges[idx])

        dE = 2.0 * spins[idx] * spin_neighbors
        dM = -2.0 * spins[idx]

        if dE < 0.0 or np.random.random() < min(np.exp(-dE / temp), 1.0):
            spins[idx] *= -1

            energy += dE
            magnet += dM

        energies[step] = energy
        magnets[step] = magnet

    return energies / N, magnets / N
