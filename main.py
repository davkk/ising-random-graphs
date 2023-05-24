import concurrent.futures
import random
from collections import defaultdict

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import typer
from igraph import Graph
from numpy.typing import NDArray


def simulate(
    *,
    steps: int,
    spins: NDArray[np.int_],
    graph: Graph,
    beta: float,
):
    print(f"{beta=:.3f}")

    neighbors = [
        np.sum(spins[graph.neighbors(i)]) for i in range(graph.vcount())
    ]

    energy = -spins @ neighbors / 2
    magnet = np.sum(spins)

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

    return beta, energy / N, magnet / N


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
    ig.config.load(".igraphrc")
    random.seed(2001)

    graph: Graph = Graph.Barabasi(n=nodes, m=m)
    spins: NDArray[np.int_] = np.random.choice([-1, 1], size=graph.vcount())
    betas: NDArray[np.float64] = np.linspace(0.01, 0.6, 100)

    data: list[tuple[float, float, float]] = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [
            executor.submit(
                simulate,
                steps=steps,
                spins=spins,
                graph=graph,
                beta=beta,
            )
            for beta in betas
            for _ in range(repeat)
        ]

        for f in concurrent.futures.as_completed(results):
            data.append(f.result())



def plot(data: list[tuple[float, float, float]]):
    _, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    averaged_E = defaultdict(list)
    for beta, avg_E, _ in data:
        averaged_E[beta].append(avg_E)
    averaged_E = dict(sorted(averaged_E.items()))

    ax_energy.scatter(
        [beta for beta, _, _ in data],
        [np.mean(avg_E) for _, avg_E, _ in data],
    )
    ax_energy.plot(
        averaged_E.keys(),
        [np.mean(energies) for energies in averaged_E.values()],
        color="orange",
    )
    ax_energy.set_title("Avg Energy")

    averaged_M = defaultdict(list)
    for beta, _, avg_M in data:
        averaged_M[beta].append(avg_M)
    averaged_M = dict(sorted(averaged_M.items()))

    ax_magnet.scatter(
        [beta for beta, _, _ in data],
        [np.mean(avg_M) for _, _, avg_M in data],
    )
    ax_magnet.plot(
        averaged_M.keys(),
        [np.mean(magents) for magents in averaged_M.values()],
        color="orange",
    )
    ax_magnet.set_title("Avg Magnetization")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    typer.run(main)
