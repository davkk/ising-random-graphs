import argparse
import math
from enum import Enum
from pathlib import Path

import networkx as nx
import numba as nb
import numpy as np

from . import estimate


def single(*, graph: nx.Graph, alpha: float):
    n = graph.number_of_nodes()
    edges = np.zeros((n, n), dtype=np.float64)

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))
        next(layers)

        for length, conn in layers:
            edges[node, conn] = np.exp(-alpha * (length - 1))

    return edges


@nb.njit(parallel=True, cache=True)
def multiple(*, graph: np.ndarray, l_max: np.int64, alpha: np.float64):
    n = graph.shape[0]
    J = np.zeros_like(graph)

    for length in nb.prange(1, l_max + 1):
        edges = np.linalg.matrix_power(graph, length)

        for i in range(n):
            for j in range(n):
                edges[i][j] *= np.exp(-alpha * (length - 1))

        J += edges

    return J


def lattice2d(*, graph: nx.Graph, alpha: float):
    n = graph.number_of_nodes()
    size = math.ceil(np.sqrt(n))
    edges = np.zeros((n, n), dtype=np.float64)

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))
        next(layers)

        x1, y1 = node
        for length, conns in layers:
            for x2, y2 in conns:
                edges[x1 + y1 * size, x2 + y2 * size] = np.exp(
                    -alpha * (length - 1)
                )

    return edges


class Method(Enum):
    single = "single"
    multiple = "multiple"
    nearest = "nearest"
    lattice2d = "lattice2d"


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
    parser.add_argument(
        "-a",
        metavar="int",
        type=np.float64,
        help="alpha parameter",
        default=1.0,
    )
    method, n, p, a = parser.parse_args().__dict__.values()

    if p and (p > 1 or p <= 0):
        parser.error("0 < p <= 1")

    graph = nx.erdos_renyi_graph(n=n, p=p)
    J, T_c = None, None

    k = (n - 1) * p

    path = Path("data/graphs/") / f"ER_{n=}_{p=}_{a=}_{method}.npy"

    match method:
        case Method.single.value:
            J = single(graph=graph, alpha=a)
            T_c = estimate.T(N=n, k=k, a=a)
            print(path, T_c)

        case Method.multiple.value:
            l_max = np.emath.logn((k - 1), (1 + n * (k - 2) / k))
            J = multiple(
                graph=nx.to_numpy_array(graph, order="C", dtype=np.float64),
                l_max=np.int64(l_max),
                alpha=a,
            )
            T_c = estimate.T(N=n, k=k, a=a)
            print(path, T_c)

        case Method.nearest.value:
            J = nx.to_numpy_array(graph)
            T_c = k
            print(path, T_c)

        case Method.lattice2d.value:
            J = lattice2d(
                graph=nx.grid_2d_graph(
                    math.ceil(np.sqrt(n)),
                    math.ceil(np.sqrt(n)),
                    periodic=True,
                ),
                alpha=a,
            )
            T_c = k
            print(path, T_c)

    np.save(path, J)


if __name__ == "__main__":
    main()
