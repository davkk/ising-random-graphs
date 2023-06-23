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
    help="CLI for running Ising simulations",
)


@app.command()
def run(
    steps: int = typer.Option(
        default=1000,
        help="Number of steps",
        min=1,
    ),
    n: int = typer.Option(
        default=64,
        help="Number of nodes",
        min=4,
    ),
    m: int = typer.Option(
        default=2,
        help="Barabasi m parameter",
        min=1,
    ),
    repeat: int = typer.Option(
        default=3,
        help="Number of repetitions for each temp",
        min=1,
    ),
    datapoints: int = typer.Option(
        default=70,
        help="Number of datapoints",
        min=1,
    ),
    temp_range: Tuple[float, float] = typer.Option(
        default=(1.0, 150.0),
        help="temp range",
        min=0.001,
    ),
    k: int = typer.Option(
        default=1,
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
        layers = np.empty((k + 1, edges.shape[0], edges.shape[1]))

        layers[0] = np.zeros_like(edges)

        for ki in range(1, k + 1):
            layers[ki] = np.linalg.matrix_power(edges, ki)

        edges = np.argmax(layers, axis=0).astype(float)
        edges[edges == 0.0] = np.inf
        edges = np.exp(-edges)

    print(f"edges:\n{edges}")

    temps: npt.NDArray[Any] = np.linspace(*temp_range, datapoints)

    sim_params = f"{n=}-{m=}-{steps=}-{repeat=}-{k=}"
    temps_filename = Path("data") / f"temps-{sim_params}.csv"
    steps_filename = Path("data") / f"steps-{sim_params}.csv"

    # TODO: use asyncio or ThreadPoolExecutor writing to a file
    with (
        open(temps_filename, "w") as temps_file,
        open(steps_filename, "w") as steps_file,
    ):
        temps_file.write("temp,energy,magnet\n")
        steps_file.write("step,temp,energy,magnet\n")

        print(f"Starting the simulation...")
        for temp, avg_E, avg_M in ising.simulate(
            steps=steps,
            init_spins=spins,
            edges=edges,
            temps=temps,
            repeat=repeat,
        ):
            temps_file.write(f"{temp:.5f},{avg_E[-1]:.5f},{avg_M[-1]:.5f}\n")
            temps_file.flush()

            for step in range(len(avg_E)):
                steps_file.write(
                    f"{step+1},{temp:.5f},{avg_E[step]:.5f},{avg_M[step]:.5f}\n"
                )
                steps_file.flush()


def save_datapoints(
    *, queue: queue.Queue, output_temps: str, output_steps: str
):
    with (
        open(output_temps, "a") as temperature_file,
        open(output_steps, "a") as steps_file,
    ):
        while True:
            try:
                item = queue.get_nowait()
                if item is None:
                    break

            except Empty:
                continue

            temp, avg_E, avg_M = item
            temperature_file.write(
                f"{temp:.5f},{avg_E[-1]:.5f},{avg_M[-1]:.5f}\n"
            )
            temperature_file.flush()

            for step in range(len(avg_E)):
                steps_file.write(
                    f"{step+1},{temp:.5f},{avg_E[step]:.5f},{avg_M[step]:.5f}\n"
                )
                steps_file.flush()


@app.command()
def plot_temps(
    filename: str = typer.Argument(Path, help="Path to data"),
):
    data = list(pd.read_csv(filename).itertuples(index=False))

    sim_params = Path(filename).stem.split("-")[1:]

    fig, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    fig.suptitle("Ising: " + ", ".join(sim_params))
    fig.canvas.manager.set_window_title("plot-temps-" + "-".join(sim_params))

    averaged_E = defaultdict(list)
    for temp, avg_E, _ in data:
        averaged_E[temp].append(avg_E)
    averaged_E = dict(sorted(averaged_E.items()))

    ax_energy.scatter(
        [temp for temp, _, _ in data],
        [np.mean(avg_E) for _, avg_E, _ in data],
        s=1,
    )
    ax_energy.plot(
        [temp for temp in averaged_E.keys()],
        [np.mean(energies) for energies in averaged_E.values()],
        color="orange",
        label="averaged",
    )
    ax_energy.set_xlabel("T")
    ax_energy.set_ylabel("<E>")

    averaged_M = defaultdict(list)
    for temp, _, avg_M in data:
        averaged_M[temp].append(avg_M)
    averaged_M = dict(sorted(averaged_M.items()))

    ax_magnet.scatter(
        [temp for temp, _, _ in data],
        [np.mean(avg_M) for _, _, avg_M in data],
        s=1,
    )
    ax_magnet.plot(
        [temp for temp in averaged_M.keys()],
        [np.mean(magents) for magents in averaged_M.values()],
        color="orange",
        label="averaged",
    )
    ax_magnet.set_xlabel("T")
    ax_magnet.set_ylabel("<M>")

    ax_energy.legend()
    ax_magnet.legend()

    plt.tight_layout()
    plt.show()


@app.command()
def plot_steps(
    filename: str = typer.Argument(Path, help="Path to data"),
):
    data = pd.read_csv(filename)
    data = list(data.itertuples(index=False))

    sim_params = Path(filename).stem.split("-")[1:]

    fig, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    fig.suptitle(", ".join(sim_params))
    fig.canvas.manager.set_window_title("plot-steps-" + "-".join(sim_params))

    ax_energy.scatter(
        [step for step, _, _, _ in data],
        [avg_E for _, _, avg_E, _ in data],
        s=1,
    )
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("<E>")

    ax_magnet.scatter(
        [step for step, _, _, _ in data],
        [avg_M for _, _, _, avg_M in data],
        s=1,
    )
    ax_magnet.set_xlabel("Step")
    ax_magnet.set_ylabel("<M>")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
