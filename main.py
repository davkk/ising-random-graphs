import random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from igraph import Graph, GraphBase
from numpy.typing import NDArray

ig.config.load(".igraphrc")


def total_energy(*, spins: NDArray[np.int8], graph: GraphBase) -> float:
    energy: float = 0

    for vertex in graph.vs:
        energy += spins[vertex.index] * np.sum(graph.neighbors(vertex.index))

    return energy / 2


def sum_neighbors(*, spins: NDArray[np.int8], neighbors: list[int]) -> float:
    sum: float = 0
    for neighbor in neighbors:
        sum += spins[neighbor]
    return sum


def simulate(
    *,
    steps: int,
    spins: NDArray[np.int8],
    graph: GraphBase,
    beta: float,
):
    energy = total_energy(spins=spins, graph=graph)
    magnet = spins.sum()

    energies = []
    magnets = []

    plt.ion()
    fig, ax = plt.subplots(2, 1)

    for _ in range(steps):
        ax[0].cla()
        ax[1].cla()

        for idx, spin in enumerate(spins):
            neighbors = sum_neighbors(
                spins=spins,
                neighbors=graph.neighbors(idx),
            )

            dE = 2 * spin * neighbors
            dM = -2 * spin

            shouldFlip = (
                np.random.random() < min(np.exp(-dE * beta), 1.0) or dE < 0
            )

            if shouldFlip:
                spins[idx] = -spins[idx]

                energy += dE
                magnet += dM

        energies.append(energy / graph.vcount())
        magnets.append(magnet / graph.vcount())

        if len(energies) > 100:
            energies.pop(0)

        if len(magnets) > 100:
            magnets.pop(0)

        ax[0].plot(energies)
        ax[1].plot(magnets)

        fig.canvas.draw()  # type: ignore
        fig.canvas.flush_events()  # type: ignore


def main():
    random.seed(2001)
    graph = Graph.Barabasi(n=1000, m=3)
    spins: NDArray[np.int8] = np.random.choice([-1, 1], size=graph.vcount())

    simulate(
        steps=int(1e7),
        spins=spins,
        graph=graph,
        beta=0.8,
    )

    input("Press enter to exit")


if __name__ == "__main__":
    import typer

    typer.run(main)
