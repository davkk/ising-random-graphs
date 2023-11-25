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
    if "critical_size_ER" in file and "single_expon" in file
]

# files = ["critical_size_ER_a=2.0_k=4_multiple_1700946756.out"]

markers = ["*", "1", "+", "2", ".", "3"]

for i, filename in enumerate(files):
    N, p, a, T_C = np.loadtxt(Path("data/processed") / filename).T

    plt.plot(
        N,
        T_C,
        "--" + markers[i],
        markersize=14,
        mew=2,
        linewidth=0.7,
        label=f"$\\alpha={a[i]}$",
    )

    plt.plot(
        N,
        estimate.T(N=N, k=(N - 1) * p, a=a),
        color="black",
    )

plt.grid(False)

plt.xlabel(r"$N$")
plt.ylabel(r"$T_C$")

plt.xscale("log")

plt.legend()

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_size.pdf", dpi=300)
plt.show()
