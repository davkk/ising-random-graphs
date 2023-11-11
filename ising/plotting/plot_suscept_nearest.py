from pathlib import Path

from ising.plotting.common import setup_pyplot
from ising.plotting.suscept import plot_suscept

setup_pyplot()


plot_suscept(
    data_dirs=[
        Path("data/processed/ising_100_ER_n=100_p=0.04_nearest/"),
        Path("data/processed/ising_100_ER_n=1000_p=0.004_nearest/"),
        Path("data/processed/ising_100_ER_n=10000_p=0.0004_nearest/"),
    ],
    title="Susceptibility $\chi$ vs. Temperature $T$, ER w/ nearest neighbor interactions",
    output_name="suscept_ER_nearest.pdf",
)
