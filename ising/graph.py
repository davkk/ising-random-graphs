import argparse
from enum import Enum

import networkx as nx
import numpy as np


def short_path(*, graph: nx.Graph):
    n = graph.number_of_nodes()
    edges = np.empty((n, n), dtype=np.float64, order="C")

    for node in graph.nodes():
        layers = enumerate(nx.bfs_layers(graph, node))

        for length, conn in layers:
            edges[node, conn] = np.exp(-length)

        edges[node, node] = 0

    return edges


def mat_power(*, graph: np.ndarray, r: np.uint8):
    if r == 1:
        return graph

    n = graph.shape[0]
    layers = np.empty((r, n, n))

    for k in range(r):
        layers[k] = np.linalg.matrix_power(graph, k + 1)
        layers[k] *= np.exp(-(k + 1))

    layers = np.sum(layers, axis=0)
    layers = layers / np.max(layers)

    return layers


class Method(Enum):
    shortpath = "shortpath"
    matpower = "matpower"


class Graph(Enum):
    ER = "ER"
    BA = "BA"


# %%
def main():
    parser = argparse.ArgumentParser(
        description="Tool to generate a random graph"
    )
    parser.add_argument(
        "graph",
        type=str,
        choices=Graph._member_names_,
        help="graph type",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=Method._member_names_,
        help="method of generating the graph",
        default=Method.shortpath,
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
        help="probability for edge creation",
    )
    parser.add_argument(
        "-m",
        metavar="int",
        type=np.uint8,
        help="number of edges to attach",
    )
    graph_type, method, n, p, m = parser.parse_args().__dict__.values()

    graph, edges, r, filename_params = None, None, None, None

    np.random.seed(2001)

    match graph_type:
        case Graph.ER.value:
            if not p:
                parser.error("The -p flag is required Erdos-Renyi graph")

            graph = nx.erdos_renyi_graph(n=n, p=p)
            r = np.ceil(np.emath.logn((n - 1) * p, n))
            filename_params = f"p={int(p * 100)}"

        case Graph.BA.value:
            if not m:
                parser.error("The -m flag is required Barabasi-Albert graph")

            graph = nx.barabasi_albert_graph(n=n, m=m)
            r = np.ceil(np.emath.logn(2 * m, n))
            filename_params = f"{m=}"

    match method:
        case Method.matpower.value:
            edges = mat_power(
                graph=nx.to_numpy_array(
                    graph,
                    dtype=np.float64,
                    order="C",
                ),
                r=r,
            )

        case Method.shortpath.value:
            edges = short_path(graph=graph)

    print(f"{edges=}")

    np.save(
        f"data/graphs/{graph_type}_{method}_{n=}_" + filename_params,
        edges,
    )


if __name__ == "__main__":
    main()
