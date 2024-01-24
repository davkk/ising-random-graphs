from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

colors, _ = common.setup_pyplot()

root = Path("data/processed") / "ising_100_ER_n=10000_p=0.0004_single"

T, _, E, M = np.loadtxt(root / "temps").T
T_avg, E_avg, M_avg = np.loadtxt(root / "temps_avg").T

fig, (energy, magnet) = plt.subplots(nrows=2, sharex=True)

energy.plot(
    T,
    E,
    "+",
    color="#cecacd",
    label="$E$ (data point)",
)
energy.plot(T_avg, E_avg, label="Average value", color=colors[3], linewidth=3)
energy.grid(True, axis="x")
energy.grid(False, axis="y")
energy.legend(loc="upper left")
energy.set_title("Energy vs. Temperature")
energy.set_ylabel(r"$E$")

magnet.plot(
    T,
    M,
    "+",
    color="#cecacd",
    label="$|M|$ (data point)",
)
magnet.plot(
    T_avg,
    M_avg,
    label="Average value",
    linewidth=3,
)
magnet.grid(True, axis="x")
magnet.grid(False, axis="y")
magnet.legend(loc="lower left")
magnet.set_title("Magnetization vs. Temperature")

magnet.set_xlabel(r"$T$")
magnet.set_ylabel(r"$|M|$")

plt.tight_layout()
plt.savefig(
    Path("figures") / "energy_magnet_vs_temp.pdf",
    dpi=300,
)
# plt.show()
