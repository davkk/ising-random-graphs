from pathlib import Path

from figures.common import plot_suscept

plot_suscept(
    data_dirs=[
        Path("data/processed/ising_100_ER_n=100_p=0.04_nearest/"),
        Path("data/processed/ising_100_ER_n=1000_p=0.004_nearest/"),
        Path("data/processed/ising_100_ER_n=10000_p=0.0004_nearest/"),
    ],
    output_name="suscept_ER_nearest.pdf",
)
