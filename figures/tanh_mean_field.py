from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import common

colors, markers = common.setup_pyplot()

xs = np.arange(-5, 5, 0.1)
tanh = np.tanh(xs)

plt.axhline(y=0, linewidth=1, color="#cecacd")
plt.axvline(x=0, linewidth=1, color="#cecacd")

plt.plot(xs, tanh, linewidth=3)

plt.plot(xs, xs, "--", label="$y=x$")
plt.plot(xs, 0.5 * xs, "--", label=r"$y=\frac{1}{2}x$")
plt.plot(xs, 2 * xs, "--", label=r"$y=2x$")

plt.grid(False)

plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.ylim(-2, 2)

plt.legend()

plt.tight_layout()
plt.savefig(Path("figures") / "tanh_mean_field.pdf", dpi=300)
# plt.show()
