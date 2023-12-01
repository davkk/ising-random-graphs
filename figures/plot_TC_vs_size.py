import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common
from ising import estimate

common.setup_pyplot()

files = [
    file
    for file in os.listdir(Path("data/processed"))
    if "critical_size_ER" in file and "single" in file
]
files.sort()

markers = ["*", "1", "+", "2", ".", "3"]

xs = np.arange(50, 100000)

for i, filename in enumerate(files):
    N, p, a, T_C = np.loadtxt(Path("data/processed") / filename).T

    dataplot = plt.plot(
        N,
        T_C,
        markers[i],
        markersize=14,
        mew=2,
        linewidth=0.7,
        label=f"$\\alpha={a[i]}$",
    )

    plt.plot(
        xs,
        estimate.T(N=xs, k=(N[i] - 1) * p[i], a=a[i]),
        "--",
        color=dataplot[0].get_color(),
        linewidth=0.7,
        label=" ",
    )

plt.grid(False)

plt.xlabel(r"$N$")
plt.ylabel(r"$T_C$")

plt.xscale("log")

plt.legend(loc="upper left", ncol=2)

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_size.pdf", dpi=300)
plt.show()
