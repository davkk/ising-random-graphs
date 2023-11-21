import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

common.setup_pyplot()

files = [
    file
    for file in os.listdir(Path("data/processed"))
    if "critical_size_ER" in file
]

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


plt.grid(False)

plt.xlabel(r"$N$")
plt.ylabel(r"$T_C$")

plt.legend()

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_size.pdf", dpi=300)
plt.show()
