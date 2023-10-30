import argparse
from enum import Enum

import networkx as nx
import numba as nb
import numpy as np

# from . import ising


def shortpath(graph: nx.Graph):
    n = graph.number_of_nodes()
    edges = np.zeros((n, n), dtype=np.float64, order="C")

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))
        next(layers)

        for length, conn in layers:
            edges[node, conn] = np.exp(-length + 1)

    return edges


@nb.njit(parallel=True, cache=True, fastmath=True)
def matpower(*, graph: np.ndarray, r: np.uint8):
    n = graph.shape[0]
    layers = np.zeros((r, n, n))

    for k in range(r):
        layers[k] = np.linalg.matrix_power(graph, k + 1)
        layers[k] *= np.exp(-k + 1)

    layers = np.sum(layers, axis=0)

    return layers / np.max(layers)


class Method(Enum):
    shortpath = "shortpath"
    matpower = "matpower"


def main():
    parser = argparse.ArgumentParser(
        description="Tool to generate a random graph"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=Method._member_names_,
        help="method of generating the graph",
        required=True,
    )
    parser.add_argument(
        "-n",
        metavar="int",
        type=np.uint16,
        required=True,
        help="number of nodes",
    )
    parser.add_argument(
        "-p",
        metavar="int",
        type=np.float64,
        required=True,
        help="probability for edge creation",
    )
    method, n, p = parser.parse_args().__dict__.values()

    if p and (p > 1 or p <= 0):
        parser.error("0 < p <= 1")

    graph = nx.erdos_renyi_graph(n=n, p=p)
    J = None

    match method:
        case Method.matpower.value:
            r_max = np.uint8(np.ceil(np.emath.logn((n - 1) * p, n)))
            J = matpower(
                graph=nx.to_numpy_array(graph, order="C"),
                r=r_max,
            )

        case Method.shortpath.value:
            # J = ising.shortpath(
            #     nx.to_numpy_array(graph, dtype=np.uint8, order="C")
            # )
            J = shortpath(graph)

    print(J)
    np.save(f"data/graphs/ER_{n=}_{p=}_{method}", J)


if __name__ == "__main__":
    main()
