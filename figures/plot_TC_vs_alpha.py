from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common
from ising import estimate

common.setup_pyplot()

alpha, T_C = np.loadtxt(
    Path("data/processed")
    / "critical_alpha_ER_n=1000_p=0.004_single_expon_1700233751"
).T

plt.grid(False)

plt.axhline(y=4, color="lightgray", linestyle="--")
plt.plot(
    alpha,
    T_C,
    "+",
    markersize=14,
    mew=2,
    linewidth=0.7,
)

xs = np.arange(1.0, 9.0, 0.01)
plt.plot(
    xs,
    estimate.T(N=1000, k=(1000 - 1) * 0.004, a=xs),
    color="#cecacd",
)

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$T_C$")

plt.xlim(1, 10)
plt.ylim(3, 10)

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_alpha.pdf", dpi=300)
plt.show()
