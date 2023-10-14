import argparse

import numpy as np
from networkx import barabasi_albert_graph, to_numpy_array
from numba import jit

np.random.seed(2001)


# @jit(nopython=True, fastmath=True, cache=True, parallel=True)
def generate(*, edges: np.ndarray, r: np.uint8):
    edges = edges.astype(np.float32)
    n = edges.shape[0]

    if r > 1:
        layers = np.empty((r, n, n), dtype=np.float32)

        for k in range(r):
            layers[k] = np.linalg.matrix_power(edges, k + 1)
            layers[k] = (layers[k] > 0.0) * (k + 1)

        layers = np.argmax(layers, 0) + 1.0
        edges = np.exp(-layers).astype(np.float32)

    return edges


def main():
    parser = argparse.ArgumentParser(
        description="Tool to generate barabasi graph"
    )
    parser.add_argument(
        "-n",
        metavar="int",
        type=np.uint16,
        required=True,
        help="number of nodes",
    )
    parser.add_argument(
        "-m",
        metavar="int",
        type=np.uint8,
        required=True,
        help="barabasi parameter",
    )
    parser.add_argument(
        "-k",
        metavar="int",
        type=np.uint8,
        required=True,
        help="range of interactions",
    )
    n, m, k = parser.parse_args().__dict__.values()

    edges = generate(
        edges=to_numpy_array(
            barabasi_albert_graph(n=n, m=m),
            dtype=np.float32,
            order="C",
        ),
        r=k,
    )

    print(edges)

    np.save(
        f"barabasi_{n=}_{m=}_{k=}",
        edges,
    )


if __name__ == "__main__":
    main()
