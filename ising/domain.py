from dataclasses import dataclass
from typing import Any

import numpy as np
from networkx import Graph, barabasi_albert_graph

from numpy.typing import NDArray


@dataclass
class Lattice:
    graph: Graph
    spins: NDArray[np.int_]
    initial_E: float
    initial_M: float

    @staticmethod
    def sum_neighbors(
        *,
        spins: NDArray[Any],
        graph: Graph,
    ) -> list[int]:
        return [
            np.sum(spins[graph.neighbors(i)]) for i in range(graph.order())
        ]

    @classmethod
    def create_ba(cls, *, n, m):
        graph: Graph = barabasi_albert_graph(n=n, m=m, seed=2001)
        spins: NDArray[Any] = np.random.choice([-1, 1], size=n)

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
