import sys

import networkx as nx
import numpy as np
import numpy.typing as npt
from numba import float64, jit

np.random.seed(2001)


@jit(nopython=True, cache=True)
def simulate(
    *,
    steps: int,
    spins: npt.NDArray[float64],
    edges: npt.NDArray[float64],
    temp: float,
):
    np.random.seed(2001)

    N = spins.shape[0]

    init_energy: float64 = -1.0 * np.sum(edges @ spins * spins) / 2.0
    init_magnet: float64 = np.sum(spins)

    energy: float64 = init_energy
    magnet: float64 = init_magnet

    for step in range(steps):
        idx: int = np.random.randint(N)

        spins[idx] *= -1.0  # change spin value
        new_energy: float64 = -1.0 * np.sum(edges @ spins * spins) / 2.0

        dE: float64 = new_energy - energy
        dM: float64 = -2.0 * spins[idx]

        if dE < 0.0 or (np.random.random() < min(np.exp(-dE / temp), 1.0)):
            energy += dE
            magnet += dM
        else:
            spins[idx] *= -1.0  # restore prev spin value, reject change

        print(temp, step, energy, magnet)


@jit(nopython=True, cache=True)
def generate(*, n: int, k: int, edges: npt.NDArray[float64]):
    spins: npt.NDArray[float64] = np.random.choice(
        np.array([-1.0, 1.0]), size=n
    )

    if k > 1:
        layers: npt.NDArray[float64] = np.empty((k, *edges.shape))

        for ki in range(k):
            layers[ki]: npt.NDArray[float64] = np.linalg.matrix_power(edges, ki + 1)

        edges: npt.NDArray[float64] = (layers != 0).argmax(axis=0)
        edges: npt.NDArray[float64] = np.where(edges != 0, np.exp(-edges), 0)

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
