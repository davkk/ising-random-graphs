import argparse
from enum import Enum
from pathlib import Path

import networkx as nx
import numba as nb
import numpy as np

from . import estimate


def single_expon(*, graph: nx.Graph, alpha: float):
    n = graph.number_of_nodes()
    edges = np.zeros((n, n), dtype=np.float64)

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))
        next(layers)

        for length, conn in layers:
            edges[node, conn] = np.exp(-alpha * (length - 1))

    return edges


def single_power(*, graph: nx.Graph, alpha: float):
    n = graph.number_of_nodes()
    edges = np.zeros((n, n), dtype=np.float64, order="C")

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))
        next(layers)

        for length, conn in layers:
            edges[node, conn] = 1 / (length**alpha)

    return edges


@nb.njit(parallel=True, cache=True)
def multiple(*, graph: np.ndarray, l_max: int, alpha: float):
    n = graph.shape[0]
    J = np.zeros((n, n))

    for length in nb.prange(1, l_max + 1):
        edges = np.linalg.matrix_power(graph, length)
        for i in range(n):
            for j in range(n):
                if edges[i][j] > 0:
                    J[i][j] += np.exp(-alpha * (length - 1))

    np.fill_diagonal(J, 0)
    J /= np.max(J)

    return J


class Method(Enum):
    single_expon = "single_expon"
    single_power = "single_power"
    multiple = "multiple"
    nearest = "nearest"


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
        case Method.single_expon.value:
            J = single_expon(graph=graph, alpha=a)
            T_c = estimate.T(N=n, k=k, a=a)
            print(path, T_c)

        case Method.single_power.value:
            J = single_power(graph=graph, alpha=a)
            T_c = estimate.T(N=n, k=k, a=a)
            print(path, T_c)

        case Method.multiple.value:
            l_max = np.emath.logn((k - 1), (1 + n * (k - 2) / k))
            J = multiple(
                graph=nx.to_numpy_array(graph, order="C", dtype=np.float64),
                l_max=int(l_max),
                alpha=a,
            )
            T_c = estimate.T(N=n, k=k, a=a)
            print(path, T_c)

        case Method.nearest.value:
            J = nx.to_numpy_array(graph)
            print(J)

    np.save(path, J)


if __name__ == "__main__":
    main()
