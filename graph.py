import argparse

import numpy as np
from networkx import barabasi_albert_graph, to_numpy_array
from numba import jit

np.random.seed(2001)


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def generate(*, graph, k):
    if k > 1:
        layers = np.empty((k, graph.shape[0], graph.shape[1]))

        for ki in range(k):
            layers[ki] = np.linalg.matrix_power(graph, ki + 1)

        graph = (layers != 0).argmax(0)
        graph = np.where(graph != 0, np.exp(-graph), 0)

    return graph


def main():
    parser = argparse.ArgumentParser(
        description="Tool to generate barabasi graph"
    )
    parser.add_argument(
        "-n",
        metavar="int",
        type=int,
        required=True,
        help="number of nodes",
    )
    parser.add_argument(
        "-m",
        metavar="int",
        type=int,
        required=True,
        help="barabasi parameter",
    )
    parser.add_argument(
        "-k",
        metavar="int",
        type=int,
        required=True,
        help="range of interactions",
    )
    n, m, k = parser.parse_args().__dict__.values()

    edges = generate(
        graph=to_numpy_array(
            barabasi_albert_graph(n=n, m=m),
            dtype=float,
        ),
        k=k,
    )

    np.save(
        f"barabasi_{n=}_{m=}_{k=}",
        edges,
    )

main()
