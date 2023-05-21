import random
from dataclasses import dataclass

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import typer
from igraph import Graph
from numpy.typing import NDArray

ig.config.load(".igraphrc")


def total_energy(*, spins: NDArray[np.int8], graph: Graph) -> float:
    energy: float = 0

    for vertex in graph.vs:  # type: ignore
        energy += spins[vertex.index] * np.sum(graph.neighbors(vertex.index))

    return energy / 2


def sum_neighbors(*, spins: NDArray[np.int8], neighbors: list[int]) -> float:
    sum: float = 0

    for neighbor in neighbors:
        sum += spins[neighbor]

    return sum


@dataclass
class Result:
    avg_energy: float
    avg_magnet: float


def simulate(
    *,
    steps: int,
    spins: NDArray[np.int8],
    graph: Graph,
    beta: float,
):
    energy = total_energy(spins=spins, graph=graph)
    magnet = spins.sum()

    N = graph.vcount()

    for _ in range(steps):
        for idx, spin in enumerate(spins):
            neighbors = sum_neighbors(
                spins=spins,
                neighbors=graph.neighbors(idx),
            )

            dE = 2 * spin * neighbors
            dM = -2 * spin

            if random.random() < min(np.exp(-dE * beta), 1) or dE < 0:
                spins[idx] = -spins[idx]

                energy += dE
                magnet += dM

    return Result(
        avg_energy=energy / N,
        avg_magnet=magnet / N,
    )


@dataclass
class DataPoint:
    beta: float
    avg_energy: np.float64
    avg_magnet: np.float64


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
):
    random.seed(2001)

    graph: Graph = Graph.Barabasi(n=nodes, m=m)
    spins: NDArray[np.int8] = np.random.choice([-1, 1], size=graph.vcount())

    data: list[DataPoint] = []

    # betas: NDArray[np.float64] = np.arange(0.01, 0.7, 0.03)
    betas: NDArray[np.float64] = np.linspace(0.01, 0.5, 100)

    for beta in betas:
        print(f"{beta=}")

        energies = []
        magnets = []

        for _ in range(100):
            result = simulate(
                steps=steps,
                spins=spins,
                graph=graph,
                beta=beta,
            )

            energies.append(result.avg_energy)
            magnets.append(result.avg_magnet)

        data.append(
            DataPoint(
                beta=beta,
                avg_energy=np.mean(energies),
                avg_magnet=np.mean(magnets),
            )
        )

    _, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    ax_energy.scatter(betas, [d.avg_energy for d in data])
    ax_magnet.scatter(betas, [d.avg_magnet for d in data])

    plt.show()


if __name__ == "__main__":
    typer.run(main)
