import random

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import typer
from igraph import Graph
from numpy.typing import NDArray

ig.config.load(".igraphrc")


def simulate(
    *,
    steps: int,
    spins: NDArray[np.int_],
    graph: Graph,
    beta: float,
):
    neighbors = [spins[graph.neighbors(i)].sum() for i in range(graph.vcount())]

    energy = -np.dot(spins, neighbors) / 2
    magnet = spins.sum()

    N = graph.vcount()

    for _ in range(steps):
        for idx, spin in enumerate(spins):
            neighbors = spins[graph.neighbors(idx)].sum()

            dE = 2 * spin * neighbors
            dM = -2 * spin

            if random.random() < min(np.exp(-dE * beta), 1) or dE < 0:
                spins[idx] = -spins[idx]

                energy += dE
                magnet += dM

    return energy / N, magnet / N


def main(
    steps: int = typer.Option(
        default=1000,
        help="Number of steps",
    ),
    nodes: int = typer.Option(
        default=100,
        help="Number of nodes",
    ),
    m: int = typer.Option(
        default=3,
        help="Barabasi m parameter",
    ),
    repeat: int = typer.Option(
        default=1,
        help="Number of repetitions for each beta",
    ),
):
    random.seed(2001)

    graph: Graph = Graph.Barabasi(n=nodes, m=m)
    spins: NDArray[np.int_] = np.random.choice([-1, 1], size=graph.vcount())
    betas: NDArray[np.float64] = np.linspace(0.01, 0.5, 20)

    data: list[DataPoint] = []

    for beta in betas:
        print(f"{beta=:.3f}")

        energies = []
        magnets = []

        for _ in range(repeat):
            avg_E, avg_M = simulate(
                steps=steps,
                spins=spins.copy(),
                graph=graph.copy(),
                beta=beta,
            )

            energies.append(avg_E)
            magnets.append(avg_M)

        data.append((beta, np.mean(energies), np.mean(magnets)))

    _, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    ax_energy.plot(
        betas,
        [avg_E for _, avg_E, _ in data],
        marker="o",
    )
    ax_magnet.plot(
        betas,
        [avg_M for _, _, avg_M in data],
        marker="o",
    )

    plt.show()


if __name__ == "__main__":
    typer.run(main)
