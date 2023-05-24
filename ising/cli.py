import concurrent.futures
import queue
import random
from collections import defaultdict
from pathlib import Path
from queue import Empty
from typing import Tuple

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from igraph import Graph
from numpy.typing import NDArray

from .ising import simulate

app = typer.Typer(
    add_completion=False,
    help="CLI for running Ising simulations",
)


def save_datapoints(*, queue: queue.Queue, output_file: str):
    with open(output_file, "a") as file:
        while True:
            try:
                item = queue.get_nowait()
                if item is None:
                    break

            except Empty:
                continue

            beta, avg_E, avg_M = item
            file.write(f"{beta:.5f},{avg_E:.5f},{avg_M:.5f}\n")
            file.flush()


@app.command()
def run(
    steps: int = typer.Option(
        default=1000,
        help="Number of steps",
    ),
    nodes: int = typer.Option(
        default=64,
        help="Number of nodes",
    ),
    m: int = typer.Option(
        default=2,
        help="Barabasi m parameter",
    ),
    repeat: int = typer.Option(
        default=3,
        help="Number of repetitions for each beta",
    ),
    datapoints: int = typer.Option(
        default=50,
        help="Number of datapoints",
    ),
    beta: Tuple[float, float] = typer.Option(
        default=(0.1, 0.6),
        help="Beta range",
    ),
):
    ig.config.load(".igraphrc")
    random.seed(2001)

    graph: Graph = Graph.Barabasi(n=nodes, m=m)
    spins: NDArray[np.int_] = np.random.choice([-1, 1], size=graph.vcount())
    betas: NDArray[np.float64] = np.linspace(*beta, datapoints)

    filename = Path("data") / f"data-{nodes=}-{m=}-{steps=}-{repeat=}.csv"

    with open(filename, "w") as f:
        f.write("beta,energy,magnet\n")

    write_queue = queue.Queue()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as writer:
        writer.submit(
            save_datapoints,
            queue=write_queue,
            output_file=filename,
        )

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
                result = f.result()
                write_queue.put_nowait(result)

            write_queue.put(None)


@app.command()
def plot(
    filename: str = typer.Argument(Path, help="Path to data"),
):
    data = list(pd.read_csv(filename).itertuples(index=False))

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
    app()
