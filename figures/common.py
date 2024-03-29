from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_pyplot():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    plt.style.use("rose-pine-dawn")

    plt.rcParams["figure.figsize"] = (9, 6)
    # plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plt.rcParams["axes.formatter.limits"] = -3, 3
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "gainsboro"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    # plt.rcParams["figure.facecolor"] = "white"
    # plt.rcParams["axes.facecolor"] = "white"
    # plt.rcParams["savefig.facecolor"] = "white"
    # plt.rcParams["figure.edgecolor"] = "black"
    # plt.rcParams["axes.edgecolor"] = "black"
    # plt.rcParams["savefig.edgecolor"] = "black"

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["*", "1", "+", "2", ".", "3"]
    return colors, markers


def plot_suscept(*, data_dirs: list[Path], output_name: str):
    setup_pyplot()

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
            xytext=(30, 25),
            textcoords="offset points",
            arrowprops=dict(
                facecolor="#286983",
                arrowstyle="->",
            ),
        )

        ax[idx].legend().set_visible(False)

        n = 100 * 10**idx
        p = 0.04 / 10**idx
        ax[idx].set_title(
            f"${n=}$, ${p=}$, $\langle k \\rangle\\approx 4$, $\\alpha=1.0$"
        )

        y_max = data[1].max()
        exponent = -int(np.floor(np.log10(y_max)))
        y_max = int(y_max * 10**exponent) + 1
        y_max /= 10**exponent
        ax[idx].set_yticks(np.linspace(0, round(y_max, exponent), 3))

    ax[0].legend().set_visible(True)

    fig.supxlabel("$T$")
    fig.supylabel("$\chi$")

    fig.tight_layout()

    plt.savefig(Path("figures") / output_name)
    # plt.show()
