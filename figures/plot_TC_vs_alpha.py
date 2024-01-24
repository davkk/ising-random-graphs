import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common
from ising import estimate

colors, markers = common.setup_pyplot()

files = [
    file
    for file in os.listdir(Path("data/processed"))
    if "critical_alpha_ER" in file and "single" in file
]
files.sort()

xs = np.arange(10, 100000)

for idx, file in enumerate(files):
    N, p, alpha, T_C = np.loadtxt(Path("data/processed") / file).T
    k = (N[idx] - 1) * p[idx]

    dataplot = plt.plot(
        alpha,
        T_C,
        markers[idx],
        markersize=14,
        mew=3,
        linewidth=0.7,
        label=f"$\\langle k\\rangle\\approx{math.ceil(k)}$",
    )

    xs = np.arange(1.0, 9.0, 0.01)
    plt.plot(
        xs,
        estimate.T(N=1000, k=k, a=xs),
        color=dataplot[0].get_color(),
        linewidth=1.5,
        label=" ",
    )

plt.legend()

plt.grid(True, "minor", "y")
plt.grid(False, "major", "x")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$T_C$")

plt.yticks([3, 4, 5, 6, 10, 15, 20])

plt.xlim(1, 9)
plt.ylim(1, 20)

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_alpha.pdf", dpi=300)
# plt.show()
