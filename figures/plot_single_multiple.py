from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

colors, markers = common.setup_pyplot()

root = Path("data/raw/single")
dirs = [
    (100, "suscept_ER_n=100_p=0.04_a=2.0_single_1701724353.out"),
    (100, "suscept_ER_n=100_p=0.04_a=2.0_multiple_1706292130.out"),
    (1000, "suscept_ER_n=1000_p=0.004_a=2.0_single_1701724636.out"),
    (
        1000,
        "suscept_ER_n=1000_p=0.004_a=2.0_multiple_1706292003.out",
    ),
    (
        10000,
        "suscept_ER_n=10000_p=0.0004_a=2.0_single_1701278332.out",
    ),
    (
        10000,
        "suscept_ER_n=10000_p=0.0004_a=2.0_multiple_1706292882.out",
    ),
]

fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True)

for idx, (n, dir_) in enumerate(dirs):
    path = root / dir_
    T, X = np.loadtxt(path).T

    x = idx // 2
    y = idx % 2

    axs[x][y].plot(
        T,
        X,
        "+",
        mew=2,
        markersize=10,
        color="#cecacd",
    )
    axs[x][0].set_title(label=f"${n=}$, BFS method")
    axs[x][1].set_title(label=f"${n=}$, powers of matrix method")

    y_max = X.min() + X.max()
    exponent = -int(np.floor(np.log10(y_max)))
    y_max = int(y_max * 10**exponent) + 1
    y_max /= 10**exponent
    axs[x][y].set_yticks(np.linspace(0, round(y_max, exponent), 3))

    axs[x][y].grid(True, axis="x")
    axs[x][y].grid(False, axis="y")


fig.supxlabel("$T$")
fig.supylabel(r"$\chi$")

plt.tight_layout()
plt.savefig(Path("figures") / "single_vs_multiple.pdf")
# plt.show()
