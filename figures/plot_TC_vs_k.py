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
    if "critical_k_ER" in file and "single" in file
]
files.sort()

xs = np.arange(0.002, 0.025, 0.0005)

for idx, file in enumerate(files):
    N, p, a, T_C = np.loadtxt(Path("data/processed") / file).T
    k = (N - 1) * p

    dataplot = plt.plot(
        k,
        T_C,
        markers[idx],
        markersize=14,
        mew=3,
        label=f"$\\alpha={a[idx]}$",
    )

    plt.plot(
        (N[idx] - 1) * xs,
        estimate.T(N=N[idx], k=(N[idx] - 1) * xs, a=a[idx]),
        color=dataplot[0].get_color(),
        linewidth=1.5,
        label=" ",
    )

plt.grid(False)

plt.xlabel(r"$\langle{k}\rangle$")
plt.ylabel(r"$T_C$")

plt.legend()

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_k.pdf", dpi=300)
# plt.show()
