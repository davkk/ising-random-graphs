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
    num_repeat: int,
):
    print(num_repeat + 1, ":", temp)

    N = spins.shape[0]

    energies = np.empty(steps)
    magnets = np.empty(steps)

    energy = -np.sum(edges @ spins * spins) / 2.0
    magnet = np.sum(spins)

    for step in range(0, steps):
        idx = np.random.randint(N)

        spins[idx] *= -1  # change spin value
        new_energy = -np.sum(edges @ spins * spins) / 2.0

        dE = new_energy - energy

        if dE < 0.0 or np.random.random() < min(np.exp(-dE / temp), 1.0):
            energy = new_energy
            magnet = np.sum(spins)
        else:
            spins[idx] *= -1  # restore prev spin value

        energies[step] = energy
        magnets[step] = magnet

    return energies / N, magnets / N
