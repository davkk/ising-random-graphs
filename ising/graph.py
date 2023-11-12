import argparse
from enum import Enum

import networkx as nx
import numba as nb
import numpy as np

from . import estimate


def single_expon(*, graph: nx.Graph, alpha: int):
    n = graph.number_of_nodes()
    edges = np.zeros((n, n), dtype=np.float64, order="C")

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))
        next(layers)

        for length, conn in layers:
            edges[node, conn] = np.exp(-alpha * (length - 1))

    return edges


def single_power(*, graph: nx.Graph, alpha: int):
    n = graph.number_of_nodes()
    edges = np.zeros((n, n), dtype=np.float64, order="C")

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))
        next(layers)

        for length, conn in layers:
            edges[node, conn] = 1 / (length**alpha)

    return edges


@nb.njit(parallel=True, cache=True)
def multiple(*, graph: np.ndarray, r_max: np.uint8, alpha: int):
    n = graph.shape[0]
    layers = np.zeros((r_max, n, n))

    for k in nb.prange(r_max):
        layers[k] = np.linalg.matrix_power(graph, k + 1)
        layers[k] *= np.exp(-alpha * (k - 1))

    layers = np.sum(layers, axis=0)

    return layers / np.max(layers)


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
        type=np.int64,
        help="alpha parameter",
        default=1,
    )
    method, n, p, a = parser.parse_args().__dict__.values()

    if p and (p > 1 or p <= 0):
        parser.error("0 < p <= 1")

    graph = nx.erdos_renyi_graph(n=n, p=p)
    J, T_c = None, None

    match method:
        case Method.single_expon.value:
            J = single_expon(graph=graph, alpha=a)
            T_c = estimate.estimate_critical_temperature(n=n, p=p, alpha=a)
            print(n, p, T_c)

        case Method.single_power.value:
            J = single_power(graph=graph, alpha=a)
            print(J)

        case Method.multiple.value:
            r_max = np.uint8(np.ceil(np.emath.logn((n - 1) * p, n)))
            J = multiple(
                graph=nx.to_numpy_array(graph, order="C"), r_max=r_max, alpha=a
            )
            print(J, f"{r_max=}")

        case Method.nearest.value:
            J = nx.to_numpy_array(graph)
            print(J)

    np.save(f"data/graphs/ER_{n=}_{p=}_{a=}_{method}", J)


if __name__ == "__main__":
    main()
