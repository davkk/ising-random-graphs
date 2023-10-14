import argparse

import numpy as np
from networkx import barabasi_albert_graph, to_numpy_array


def generate(*, edges: np.ndarray, r: np.uint8):
    if r == 1:
        return edges

    n = edges.shape[0]
    layers = np.empty((r, n, n))

    for k in range(r):
        layers[k] = np.linalg.matrix_power(edges, k + 1)
        layers[k] *= np.exp(-(k + 1))

    layers = np.sum(layers, axis=0)
    layers = layers / np.max(layers)

    return layers


# %%
def main():
    parser = argparse.ArgumentParser(
        description="Tool to generate a Barabasi-Albert graph"
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
        "-r",
        metavar="int",
        type=np.uint8,
        required=True,
        help="range of interactions",
    )
    n, m, r = parser.parse_args().__dict__.values()

    np.random.seed(2001)

    edges = generate(
        edges=to_numpy_array(
            barabasi_albert_graph(n=n, m=m),
            dtype=np.float64,
            order="C",
        ),
        r=r,
    )

    print(f"{edges=}")

    np.save(
        f"barabasi_{n=}_{m=}_{r=}",
        edges,
    )


if __name__ == "__main__":
    main()
