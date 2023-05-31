from dataclasses import dataclass

import numpy as np
from igraph import Graph
from numpy.typing import NDArray


@dataclass
class Parameters:
    steps: int
    beta: float


@dataclass
class Lattice:
    graph: Graph
    spins: NDArray[np.int_]
    initial_E: float
    initial_M: float

    @staticmethod
    def sum_neighbors(
        *,
        spins: NDArray[np.int_],
        graph: Graph,
    ) -> list[int]:
        return [
            np.sum(spins[graph.neighbors(i)]) for i in range(graph.vcount())
        ]

    @classmethod
    def create_ba(cls, *, n, m):
        graph: Graph = Graph.Barabasi(n=n, m=m)
        spins: NDArray[np.int_] = np.random.choice([-1, 1], size=n)

        energy = -spins @ cls.sum_neighbors(spins=spins, graph=graph) / 2
        magnet = np.sum(spins)

        return cls(
            graph=graph,
            spins=spins,
            initial_E=energy,
            initial_M=magnet,
        )

    def copy(self):
        return Lattice(
            graph=self.graph.copy(),
            spins=self.spins.copy(),
            initial_E=self.initial_E,
            initial_M=self.initial_M,
        )
