from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_suscept(*, data_dirs: list[str], title: str, output_name: str):
    fig, ax = plt.subplots(len(data_dirs), 1, sharex=True)

    for idx, data_dir in enumerate(data_dirs):
        data = np.loadtxt(data_dir / "suscept").T
        data_avg = np.loadtxt(data_dir / "suscept_avg").T

        max_idx = data_avg.argmax(axis=1)[1]
        max_x, max_y = data_avg[0][max_idx], data_avg[1][max_idx]

        ax[idx].axhline(linewidth=0.5, color="#797593")
        ax[idx].axvline(x=max_x, linestyle="dashed", color="#797593")

        ax[idx].scatter(
            list(data[0]),
            list(data[1]),
            label="Susceptibility $\chi$",
            marker="+",
            c="#cecacd",
        )
        ax[idx].plot(
            list(data_avg[0]),
            list(data_avg[1]),
            label="Average value",
            linewidth=3,
        )
        ax[idx].margins(0, 0.1)

        ax[idx].annotate(
            f"$T_C\\approx{max_x}$",
            xy=(max_x, max_y),
            xytext=(30, 30),
            textcoords="offset points",
            arrowprops=dict(
                facecolor="#286983",
                arrowstyle="->",
            ),
        )

        ax[idx].legend().set_visible(False)

        n = 100 * 10**idx
        p = 0.04 / 10**idx
        ax[idx].set_title(f"${n=}$, ${p=}$, $\langle k \\rangle\\approx 4$")

    ax[0].legend().set_visible(True)

    fig.suptitle(title)
    fig.supxlabel("$T$")
    fig.supylabel("$\chi$")

    fig.tight_layout()

    plt.savefig(Path("figures") / output_name)
