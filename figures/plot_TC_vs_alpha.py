from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

common.setup_pyplot()

alpha, T_C = np.loadtxt(
    Path("data/processed") / "critical_ER_n=1000_p=0.004_single_expon_alpha"
).T

plt.plot(alpha, T_C, "-o")

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$T_C$")

plt.margins(0.1, 0.3)

# TODO: extrapolate?

plt.tight_layout()
plt.savefig(Path("figures") / "TC_vs_alpha_log.pdf", dpi=300)
plt.show()
