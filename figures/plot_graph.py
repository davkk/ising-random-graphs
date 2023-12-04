from pathlib import Path

import common
import networkx as nx
from matplotlib import pyplot as plt

colors, _ = common.setup_pyplot()


fig, axs = plt.subplots(ncols=3, nrows=2)

for x in range(2):
    for y in range(3):
        p = 0.01 * (x * 3 + y + 2)
        nx.draw_kamada_kawai(
            nx.erdos_renyi_graph(n=100, p=p, seed=2001),
            ax=axs[x][y],
            node_size=70,
            node_color=colors[(x * 3 + y) % len(colors)],
            edge_color="#babebd",
        )

        axs[x][y].set_title(f"$N=100$, ${p=}$")

fig.tight_layout()

# now = int(datetime.now().timestamp())
plt.savefig(Path("figures") / "ER_graph.pdf")
plt.show()
