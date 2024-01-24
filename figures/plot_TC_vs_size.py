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
    if "critical_size_ER" in file and "single" in file
]
files.sort()

xs = np.arange(50, 100000, 100)
k = 0

for i, filename in enumerate(files):
    N, p, a, T_C = np.loadtxt(Path("data/processed") / filename).T
    k = (N[i] - 1) * p[i]

    dataplot = plt.plot(
        N,
        T_C,
        markers[i],
        markersize=14,
        mew=3,
        linewidth=0.7,
        label=f"$\\alpha={a[i]}$",
    )

    plt.plot(
        xs,
        estimate.T(N=xs, k=k, a=a[i]),
        color=dataplot[0].get_color(),
        linewidth=1.5,
        label=" ",
    )

a_boundary = np.log(k - 1) + 1e-10
plt.plot(
    xs,
    estimate.T(N=xs, k=k, a=a_boundary),
    "--",
    color="#cecacd",
    linewidth=1.7,
    label=f"$\\alpha_C={a_boundary:.5f}$",
)


plt.grid(False)

plt.xlabel(r"$n$")
plt.ylabel(r"$T_C$")

plt.xscale("log")

plt.legend(loc="upper left", ncol=1)

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_size.pdf", dpi=300)
# plt.show()
