from pathlib import Path

from figures.common import plot_suscept, setup_pyplot

setup_pyplot()


plot_suscept(
    data_dirs=[
        Path("data/processed/ising_100_ER_n=100_p=0.04_single/"),
        Path("data/processed/ising_100_ER_n=1000_p=0.004_single/"),
        Path("data/processed/ising_100_ER_n=10000_p=0.0004_single/"),
    ],
    title="Susceptibility $\chi$ vs. Temperature $T$, ER w/ single long-range interactions",
    output_name="suscept_ER_single.pdf",
)
