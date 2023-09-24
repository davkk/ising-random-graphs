from pathlib import Path
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
        default=(0.4, 40.0),
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

        edges = (layers != 0).argmax(axis=0)
        edges = np.where(edges != 0, np.exp(-edges), 0)

    print(f"edges:\n{edges}")

    temps: npt.NDArray[Any] = np.linspace(*temp_range, datapoints)

    sim_params = f"{n=}-{m=}-{steps=}-{repeat=}-{k=}"
    output_path = Path("data") / f"data-{sim_params}.csv"

    with open(output_path, "w") as file:
        file.write("temp,step,energy,magnet\n")

        print(f"Starting the simulation...")

        for temp, step, avg_E, avg_M in ising.simulate(
            steps=steps,
            init_spins=spins,
            edges=edges,
            temps=temps,
            repeat=repeat,
        ):
            file.write(f"{temp:.5f},{step+1},{avg_E:.5f},{avg_M:.5f}\n")

            if step % 100 == 0:
                file.flush()


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
    data = pd.read_csv(filename).groupby("step", as_index=False)

    data_max = data.max()
    data_min = data.min()

    sim_params = Path(filename).stem.split("-")[1:]

    fig, (ax_energy, ax_magnet) = plt.subplots(nrows=2, ncols=1)

    fig.suptitle(", ".join(sim_params))
    fig.canvas.manager.set_window_title("plot-steps-" + "-".join(sim_params))

    data_max.plot.scatter(
        ax=ax_energy,
        x="step",
        y="energy",
        s=1,
        label=f"$T_\max={data_max.temp[0]}$",
        color="orange",
    )
    data_min.plot.scatter(
        ax=ax_energy,
        x="step",
        y="energy",
        s=1,
        label=f"$T_\min={data_min.temp[0]}$",
    )
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("<E>")

    data_max.plot.scatter(
        ax=ax_magnet,
        x="step",
        y="magnet",
        s=1,
        label=f"$T_\max={data_max.temp[0]}$",
        color="orange",
    )
    data_min.plot.scatter(
        ax=ax_magnet,
        x="step",
        y="magnet",
        s=1,
        label=f"$T_\min={data_min.temp[0]}$",
    )
    ax_magnet.set_xlabel("Step")
    ax_magnet.set_ylabel("<M>")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
