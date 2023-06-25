import concurrent.futures
import os
import queue
import sys
from collections import defaultdict
from pathlib import Path
from queue import Empty
from typing import Any, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import typer
from numpy import typing as npt

from . import ising

app = typer.Typer(
    add_completion=False,
    help="CLI for running Ising simulations on Barabasi-Albert networks",
)


def save_datapoints(*, queue: queue.Queue, path: str):
    with open(path, "a") as file:
        while True:
            try:
                item = queue.get_nowait()
                if item is None:
                    break

            except Empty:
                continue

            temp, avg_E, avg_M = item

            for step in range(len(avg_E)):
                file.write(
                    f"{temp:.5f},{step+1},{avg_E[step]:.5f},{avg_M[step]:.5f}\n"
                )
                file.flush()


@app.command()
def run(
    steps: int = typer.Option(
        default=1000,
        help="Number of steps",
        min=1,
    ),
    n: int = typer.Option(
        default=128,
        help="Number of nodes",
        min=4,
    ),
    m: int = typer.Option(
        default=3,
        help="Barabasi m parameter",
        min=1,
    ),
    repeat: int = typer.Option(
        default=5,
        help="Number of repetitions for each temp",
        min=1,
    ),
    datapoints: int = typer.Option(
        default=100,
        help="Number of datapoints",
        min=1,
    ),
    temp_range: Tuple[float, float] = typer.Option(
        default=(1.0, 100.0),
        help="temp range",
        min=0.001,
    ),
    k: int = typer.Option(
        default=3,
        help="Range of interactions",
        min=1,
        max=8,
    ),
):
    np.random.seed(2001)

    graph: nx.Graph = nx.barabasi_albert_graph(n=n, m=m)
    edges = nx.to_numpy_array(graph, dtype=float)
    spins: npt.NDArray[Any] = np.random.choice([-1.0, 1.0], size=n)

    if k > 1:
        layers = np.empty((k, *edges.shape))

        for ki in range(k):
            layers[ki] = np.linalg.matrix_power(edges, ki + 1)

        edges = np.argmax(layers, axis=0)
        edges = np.exp(-edges)
        edges[np.sum(layers, axis=0) == 0] = 0

    print(f"edges:\n{edges}")

    temps: npt.NDArray[Any] = np.linspace(*temp_range, datapoints)

    sim_params = f"{n=}-{m=}-{steps=}-{repeat=}-{k=}"
    output_path = Path("data") / f"data-{sim_params}.csv"

    write_queue = queue.Queue()

    with open(output_path, "w") as output_file:
        output_file.write("temp,step,energy,magnet\n")

    with concurrent.futures.ThreadPoolExecutor(1) as writer:
        writer.submit(
            save_datapoints,
            queue=write_queue,
            path=output_path,
        )

        print(f"Starting the simulation...")

        for result in ising.simulate(
            steps=steps,
            init_spins=spins,
            edges=edges,
            temps=temps,
            repeat=repeat,
        ):
            write_queue.put_nowait(result)

        write_queue.put(None)


@app.command()
def plot_graph(
    n: int = typer.Option(
        int,
        help="Number of nodes",
        min=4,
    ),
    m: int = typer.Option(
        int,
        help="Barabasi m parameter",
        min=1,
    ),
):
    graph: nx.Graph = nx.barabasi_albert_graph(n=n, m=m)
    nx.draw(graph)
    plt.show()


@app.command()
def plot_temps(
    filename: str = typer.Argument(Path, help="Path to data file"),
):
    data = pd.read_csv(filename)
    data = data[data.step == data.max().step]
    averaged = data.groupby("temp", as_index=False).mean()

    sim_params = Path(filename).stem.split("-")[1:]

    fig, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    fig.suptitle("Ising: " + ", ".join(sim_params))
    fig.canvas.manager.set_window_title("plot-temps-" + "-".join(sim_params))

    ax_energy.scatter(
        data.temp.tolist(),
        data.energy.tolist(),
        s=1,
    )
    ax_energy.plot(
        averaged.temp.tolist(),
        averaged.energy.tolist(),
        color="orange",
        label="averaged",
    )
    ax_energy.set_xlabel("T")
    ax_energy.set_ylabel("<E>")

    ax_magnet.scatter(
        data.temp.tolist(),
        data.magnet.tolist(),
        s=1,
    )
    ax_magnet.plot(
        averaged.temp.tolist(),
        averaged.magnet.tolist(),
        color="orange",
        label="averaged",
    )
    ax_magnet.set_xlabel("T")
    ax_magnet.set_ylabel("<E>")

    ax_energy.legend()
    ax_magnet.legend()

    plt.tight_layout()
    plt.show()


@app.command()
def plot_steps(
    filename: str = typer.Argument(Path, help="Path to data"),
):
    data = pd.read_csv(filename)

    sim_params = Path(filename).stem.split("-")[1:]

    fig, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    fig.suptitle(", ".join(sim_params))
    fig.canvas.manager.set_window_title("plot-steps-" + "-".join(sim_params))

    data.plot.scatter(
        ax=ax_energy,
        x="step",
        y="energy",
        s=1,
    )
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("<E>")

    data.plot.scatter(
        ax=ax_magnet,
        x="step",
        y="magnet",
        s=1,
    )
    ax_magnet.set_xlabel("Step")
    ax_magnet.set_ylabel("<M>")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
