import sys
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import float64, jit

np.random.seed(2001)


@jit(nopython=True, cache=True)
def simulate(
    *,
    steps: float,
    spins: npt.NDArray[Any],
    edges: npt.NDArray[Any],
    temp: float,
):
    np.random.seed(2001)

    N = spins.shape[0]

    init_energy: float = -np.sum(edges @ spins * spins) / 2.0
    init_magnet: float = np.sum(spins)

    energy = init_energy
    magnet = init_magnet

    for step in range(steps):
        idx = np.random.randint(N)

        spins[idx] *= -1  # change spin value
        new_energy = -np.sum(edges @ spins * spins) / 2.0

        dE = new_energy - energy
        dM = -2 * spins[idx]

        if dE < 0.0 or (np.random.random() < min(np.exp(-dE / temp), 1.0)):
            energy += dE
            magnet += dM
        else:
            spins[idx] *= -1  # restore prev spin value, reject change

        print(temp, step, energy, magnet)


@jit(nopython=True, cache=True)
def generate(*, n: int, k: int, edges: npt.NDArray[float64]):
    spins: npt.NDArray[float64] = np.random.choice(
        np.array([-1.0, 1.0]), size=n
    )

    if k > 1:
        layers: npt.NDArray[float64] = np.empty((k, *edges.shape))

        for ki in range(k):
            layers[ki] = np.linalg.matrix_power(edges, ki + 1)

        edges = (layers != 0).argmax(axis=0)
        edges = np.where(edges != 0, np.exp(-edges), 0)

    return spins, edges


temp = float(sys.argv[1])
steps = 100_000
n = 128
m = 3
k = 3

spins, edges = generate(
    n=n,
    k=k,
    edges=nx.to_numpy_array(
        nx.barabasi_albert_graph(n=n, m=m),
        dtype=float,
    ),
)

simulate(
    temp=temp,
    steps=steps,
    spins=spins,
    edges=edges,
)
