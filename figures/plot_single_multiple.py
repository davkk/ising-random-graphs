from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

colors, markers = common.setup_pyplot()

root = Path("data/raw/single")
dirs = [
    (100, "single", "suscept_ER_n=100_p=0.04_a=2.0_single_1701724353.out"),
    (100, "multiple", "suscept_ER_n=100_p=0.04_a=2.0_multiple_1701723855.out"),
    (1000, "single", "suscept_ER_n=1000_p=0.004_a=2.0_single_1701724636.out"),
    (
        1000,
        "multiple",
        "suscept_ER_n=1000_p=0.004_a=2.0_multiple_1701724567.out",
    ),
    (
        10000,
        "single",
        "suscept_ER_n=10000_p=0.0004_a=2.0_single_1701278332.out",
    ),
    (
        10000,
        "multiple",
        "suscept_ER_n=10000_p=0.0004_a=2.0_multiple_1701723477.out",
    ),
]

fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True)

for idx, (N, model, dir_) in enumerate(dirs):
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
    axs[x][y].set_title(label=f"${N=}$, {model} interaction")

    idx == 0 and axs[x][y].legend()

    axs[x][y].grid(True, axis="x")
    axs[x][y].grid(False, axis="y")


fig.supxlabel("$T$")
fig.supylabel("$X$")

plt.tight_layout()
plt.savefig(Path("figures") / "single_vs_multiple.pdf")
plt.show()
