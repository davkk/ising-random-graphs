from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common
from ising import estimate

common.setup_pyplot()

filename = "critical_k_ER_n=1000_a=2.0_single_1701203782.out"

markers = ["*", "1", "+", "2", ".", "3"]

N, p, a, T_C = np.loadtxt(Path("data/processed") / filename).T
k = (N - 1) * p

plt.plot(
    k,
    T_C,
    "+",
    markersize=14,
    mew=2,
    linewidth=0.7,
    label="simulation points",
)

plt.plot(
    k,
    estimate.T(N=N, k=k, a=a),
    "--",
    color="#cecacd",
    label="analytical approximation"
)

plt.grid(False)

plt.xlabel(r"$\langle{k}\rangle$")
plt.ylabel(r"$T_C$")

plt.legend()

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_k.pdf", dpi=300)
plt.show()
