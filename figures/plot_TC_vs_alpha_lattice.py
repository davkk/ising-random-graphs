from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

colors, markers = common.setup_pyplot()

file = "critical_alpha_lattice_n=1000_p=0.004_single_1702742344.out"

xs = np.arange(10, 100000)

N, p, alpha, T_C = np.loadtxt(Path("data/processed") / file).T
k: int = (N[0] - 1) * p[0]

avg_T = np.average(T_C)

plt.axhline(
    y=avg_T,
    c="#cecacd",
    linewidth=3,
)

plt.plot(
    alpha,
    T_C,
    markers[0],
    markersize=14,
    mew=3,
    linewidth=0.7,
)

plt.grid(True, "minor", "y")
plt.grid(False, "major", "x")

plt.yticks([0, 1, 2, avg_T, 3, 4, 5])

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$T_C$")

plt.ylim(0, 4)

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_alpha_lattice.pdf", dpi=300)
# plt.show()
