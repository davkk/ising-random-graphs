import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

colors, markers = common.setup_pyplot()

root = Path("data/raw") / "ising_100_ER_n=10000_p=0.0004_single"

temps = [20, 40, 60, 80, 100]

for idx, temp in enumerate(temps):
    path = root / f"{temp:.1f}"
    files = os.listdir(path)

    _, steps, E, M = np.loadtxt(path / "1").T
    plt.plot(
        steps,
        E,
        markers[idx + 1],
        mew=2,
        markersize=10,
        color=colors[idx],
        linewidth=1.2,
        label=f"$T={temp}$",
    )

plt.legend(loc="lower right")

plt.xlabel("$t$")
plt.ylabel("$M$")

plt.tight_layout()
plt.savefig(Path("figures") / "magnet_vs_steps.pdf")
plt.show()
