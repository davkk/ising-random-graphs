from pathlib import Path

from ising.plotting.common import plot_suscept, setup_pyplot

setup_pyplot()


plot_suscept(
    data_dirs=[
        Path("data/processed/ising_70_ER_n=100_p=0.04_multiple/"),
        Path("data/processed/ising_70_ER_n=1000_p=0.004_multiple/"),
        Path("data/processed/ising_70_ER_n=10000_p=0.0004_multiple/"),
    ],
    title="Susceptibility $\chi$ vs. Temperature $T$, ER w/ multiple long-range interactions",
    output_name="suscept_ER_multiple.pdf",
)
