import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def simulate(
    *,
    steps: int,
    init_spins: npt.NDArray[Any],
    edges: npt.NDArray[Any],
    temps: npt.NDArray[Any],
    repeat: int,
):
    N = init_spins.shape[0]

    for temp in temps:
        for num_repeat in range(repeat):
            print(num_repeat + 1, ":", temp)

            spins = init_spins.copy()

            energies = np.empty(steps)
            magnets = np.empty(steps)

            energy = -np.sum(edges @ spins * spins) / 2.0
            magnet = np.sum(spins)

            for step in range(0, steps):
                idx = np.random.randint(N)

                spins[idx] *= -1  # change spin value
                new_energy = -np.sum(edges @ spins * spins) / 2.0

                dE = new_energy - energy
                dM = -2 * spins[idx]

                if dE < 0.0 or (
                    np.random.random() < min(np.exp(-dE / temp), 1.0)
                ):
                    energy += dE
                    magnet += dM
                else:
                    spins[idx] *= -1  # restore prev spin value

                energies[step] = energy
                magnets[step] = magnet

            yield temp, energies / N, magnets / N
